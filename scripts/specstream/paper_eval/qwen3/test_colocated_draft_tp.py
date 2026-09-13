"""CPU-only regression tests for the actual launch command builder (no models)."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class DraftPlacementTests(unittest.TestCase):
    def run_case(self, method, draft_tp=None, target_tp=2):
        repo = Path(os.environ.get('REPO', Path(__file__).resolve().parents[4]))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        env = dict(os.environ, REPO=str(repo), SPECSTREAM_DRY_RUN='1',
                   SPECSTREAM_PYTHON=os.environ.get('SPECSTREAM_PYTHON', 'python'),
                   TARGET_MODEL='/models/target', DRAFT_MODEL='/models/draft',
                   TARGET_UUID='GPU-test0', COLOCATED_UUID='GPU-test0', DRAFT_UUID='GPU-test0',
                   TARGET_UUIDS='GPU-test0,GPU-test1' if target_tp == 2 else 'GPU-test0',
                   TARGET_TP_SIZE=str(target_tp), COLOCATED_TP_RANK='0',
                   METHOD=method, DATASET_TAG='placement', DATASET_NAME='random-ids',
                   INPUT_LEN='32000', NUM_PROMPTS='64', OUTPUT_LEN='256', MAX_CONCURRENCY='16',
                   RESULT_ROOT=str(root), SPECSTREAM_DRAFT_MAX_TOTAL_TOKENS='516224',
                   SPECSTREAM_DRAFT_MIN_KV_TOKENS='516224', SPECSTREAM_OVERLAP_MODE='auto',
                   SPECSTREAM_FIXED_Q='0')
        if draft_tp is None:
            env.pop('SPECSTREAM_DRAFT_TP_SIZE', None)
        else:
            env['SPECSTREAM_DRAFT_TP_SIZE'] = str(draft_tp)
        result = subprocess.run(['bash', str(repo / 'scripts/specstream/paper_eval/qwen3/run_public_once.sh')],
                                cwd=repo, env=env, text=True, capture_output=True, timeout=30)
        if draft_tp == 3:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Draft TP must be', result.stdout + result.stderr)
            return
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        case = root / 'logs' / f'{method}_placement_c16'
        cfg = dict(x.split('=', 1) for x in (case / 'config.env').read_text().splitlines() if '=' in x)
        expected_tp = draft_tp or target_tp
        self.assertEqual(cfg['DRAFT_TP_SIZE'], str(expected_tp))
        self.assertEqual(cfg['DRAFT_VISIBLE'], cfg['TARGET_VISIBLE'] if expected_tp > 1 else 'GPU-test0')
        self.assertEqual(cfg['FIXED_Q_MODE'], 'ordinary')
        self.assertEqual(cfg['SPECSTREAM_OVERLAP_MODE'], 'serial')
        cmd = shlex.split((case / 'draft_command.txt').read_text())
        self.assertEqual(cmd.count('--tp-size'), 1)
        self.assertEqual(cmd[cmd.index('--tp-size') + 1], str(expected_tp))
        self.assertEqual(cmd[cmd.index('--max-total-tokens') + 1], '516224')
        self.assertFalse((case / 'case_complete.marker').exists())
        self.assertFalse((case / 'tp_windows').exists())
        if method.startswith('K'):
            self.assertEqual(cfg['VERIFY_Q'], '4' if method in ['K1', 'K2', 'K3'] else 'dynamic_max_8')
            self.assertEqual(cfg['SERIALIZE_H2D'], '1' if method in ['K1', 'K2'] else '0')

    def test_all_colocated_ordinary_methods_follow_target(self):
        for method in ['K1', 'K2', 'K3', 'K4', 'K5', 'I1_GPU_ONLY', 'I1_SEALED_HISTORY', 'SGLANG_SD_KV_OFFLOAD', 'A']:
            with self.subTest(method=method):
                self.run_case(method)

    def test_explicit_tp1_placement_control(self):
        self.run_case('K3', draft_tp=1)

    def test_single_target_gpu(self):
        self.run_case('K3', target_tp=1)

    def test_reject_partial_tp_group(self):
        self.run_case('K3', draft_tp=3)


if __name__ == '__main__':
    unittest.main()
