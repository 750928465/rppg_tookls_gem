"""Log-monitor evidence boundaries; no training or real dataset access."""
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/inspect_bbpnet_progress.py'
spec = importlib.util.spec_from_file_location('bbp_progress', SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def line(epoch, score=8.0, n_states=173):
    valid = dict(loss=4.0, sbp_mae=score + 1, dbp_mae=score - 1, n_states=n_states)
    return f"STAGE2 epoch={epoch} train={{'loss': 3.5}} valid={valid!r}\n"


def assessment(scores, **kwargs):
    rows, issues, _ = monitor.parse_log(''.join(line(i, s) for i, s in enumerate(scores)))
    return monitor.assess(rows, issues, **kwargs)


class ProgressTests(unittest.TestCase):
    def test_active_tqdm_ansi_and_incomplete_epoch(self):
        text = ('STAGE2 epoch 1/100; lr=0.001\n\x1b[32mSTAGE2 train: 50%\x1b[0m\r'
                + line(0) + 'STAGE2 epoch 2/100; lr=0.00099\n'
                + 'STAGE2 valid: 25%| 100/4569\r'
                + "STAGE2 epoch=1 train={'loss': 2} valid={")
        rows, issues, progress = monitor.parse_log(text)
        self.assertEqual([r['epoch'] for r in rows], [1])
        self.assertEqual(rows[0]['checkpoint_index'], 0)
        self.assertEqual(issues, [])
        self.assertEqual(progress['latest_epoch_header'], '2/100')
        self.assertIn('valid: 25%', progress['latest_progress'])
        self.assertEqual(monitor.assess(rows, issues)['status'], 'INSUFFICIENT_HISTORY')

    def test_plateau_needs_minimum_history_and_patience(self):
        self.assertEqual(assessment([8.] * 19)['status'], 'INSUFFICIENT_HISTORY')
        result = assessment([8.] * 20)
        self.assertEqual(result['status'], 'PLATEAU_CANDIDATE')
        self.assertEqual(result['best']['epoch'], 1)
        self.assertEqual(result['epochs_without_significant_improvement'], 19)
        # A recent improvement resets patience even after many plateau epochs.
        self.assertEqual(assessment([8.] * 25 + [7.5])['status'], 'KEEP_OBSERVING')

    def test_continued_small_improvements_accumulate(self):
        result = assessment([10. - i * .03 for i in range(30)])
        self.assertEqual(result['status'], 'KEEP_OBSERVING')
        self.assertLess(result['epochs_without_significant_improvement'], 4)
        self.assertEqual(result['best']['epoch'], 30)

    def test_threshold_equality_and_actual_best_are_distinct(self):
        result = assessment([8., 7.9, 7.85], min_epochs=1, patience=1)
        self.assertEqual(result['last_significant_improvement_epoch'], 2)
        self.assertEqual(result['best']['epoch'], 3)
        self.assertEqual(result['epochs_without_significant_improvement'], 1)

    def test_partial_capture_or_missing_epoch_cannot_recommend_stopping(self):
        for indices in (range(5, 30), [i for i in range(30) if i != 4]):
            rows, issues, _ = monitor.parse_log(''.join(line(i) for i in indices))
            self.assertEqual(monitor.assess(rows, issues)['status'], 'UNRELIABLE_HISTORY')

    def test_nonfinite_metric_invalidates_assessment(self):
        text = ''.join(line(i) for i in range(21)) + line(21).replace("'loss': 4.0", "'loss': nan")
        rows, issues, _ = monitor.parse_log(text)
        self.assertEqual(monitor.assess(rows, issues)['status'], 'UNRELIABLE_HISTORY')

    def test_validation_population_change_is_flagged(self):
        text = ''.join(line(i) for i in range(20)) + line(20, n_states=170)
        rows, issues, _ = monitor.parse_log(text)
        self.assertEqual(monitor.assess(rows, issues)['status'], 'UNRELIABLE_HISTORY')

    def test_mixed_runs_are_not_merged(self):
        with self.assertRaisesRegex(ValueError, '重复或倒退'):
            monitor.parse_log(line(0) + line(1) + line(0))

    def test_parameters_reject_nan_or_invalid_counts(self):
        for kwargs in ({'min_epochs': 0}, {'patience': 0}, {'min_delta': -1}, {'min_delta': float('nan')}):
            with self.assertRaises(ValueError):
                assessment([8.], **kwargs)


if __name__ == '__main__':
    unittest.main()
