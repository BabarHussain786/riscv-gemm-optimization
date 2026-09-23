"""Host-only orchestration tests. No kernels or real experiments are executed."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import paper_campaign as pc


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = pc.HERE.parent
        cls.args = pc.arguments([])
        cls.kernels, cls.coverage = pc.inventory(cls.root)
        cls.cases = pc.plan_cases(cls.kernels, cls.args)

    def test_inventory_and_unroll_coverage(self):
        self.assertEqual(len(self.kernels), 108)
        self.assertEqual(len({c['id'] for c in self.cases}), len(self.cases))
        for kind, tile, lmul in {(k['kind'], k['tile'], k['lmul']) for k in self.kernels}:
            values = {k['unroll'] for k in self.kernels if (k['kind'],k['tile'],k['lmul']) == (kind,tile,lmul)}
            self.assertEqual(values, {1,2,4,8})
        self.assertFalse(any(k['kind']=='INT8_RVV' and k['lmul'] in ('4','8') for k in self.kernels))
        self.assertTrue(any(k['status']=='experimental_opt_in_required' for k in self.coverage))

    def test_all_figures_and_weak_sizes(self):
        self.assertEqual({c['figure'] for c in self.cases}, set(pc.PLOTS.values()))
        weak = [c for c in self.cases if c['figure']==pc.PLOTS[2]]
        self.assertEqual({tuple(c['dimensions']) for c in weak}, {(n,n,n) for n in (512,672,832,1024)})
        self.assertTrue(all(c['threads']<=4 for c in weak if c['kind']=='INT8_IME'))

    def test_modes_and_scope(self):
        mixed = [c for c in self.cases if c['mode']=='k1-mixed-rvv-ime']
        self.assertTrue(all(c['kind']=='INT8_IME' and c['lmul']=='1' for c in mixed))
        self.assertEqual({c['policy'] for c in mixed}, {'static','dynamic'})
        tuning = [c for c in self.cases if c['figure']==pc.PLOTS[4]]
        self.assertTrue(all(c['runner']=='standalone' and c['kind']=='INT8_RVV' for c in tuning))

    def test_environment_isolation(self):
        with patch.dict('os.environ', {'KERNEL_FILTER':'bad','OUT_DIR':'old','GEMM_VALIDATE':'0'}):
            cmd, env = pc.command_for(self.cases[0], self.root, Path('/fresh'), self.args)
        self.assertEqual(env['GEMM_VALIDATE'], '1')
        self.assertEqual(env['OUT_DIR'], str(Path('/fresh')))
        self.assertNotEqual(env['KERNEL_FILTER'], 'bad')
        self.assertEqual(cmd[0], 'bash')

    def test_failure_classification(self):
        self.assertEqual(pc.classify(0, []), 'missing_data')
        self.assertEqual(pc.classify(1, [{'status':'OK'}]), 'failed')
        self.assertEqual(pc.classify(0, [{'status':'BUILD_FAILED'}]), 'failed_validation_or_run')
        self.assertEqual(pc.classify(0, [{'status':'OK'}]), 'complete')
        self.assertEqual(pc.classify(0, [{'status':'OK'}], 6), 'incomplete_sample_count')

    def test_dry_run_and_resume(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder)/'campaign'
            options = ['--dry-run','--output',str(output),'--figures','1']
            with patch('subprocess.Popen', side_effect=AssertionError('must not execute')):
                self.assertEqual(pc.main(options), 0)
                self.assertEqual(pc.main(options+['--resume']), 0)
                with self.assertRaises(SystemExit):
                    pc.main(options+['--resume','--runs','2'])
            self.assertFalse(list(output.rglob('raw_data.csv')))

    def test_aggregation_preserves_failures(self):
        # Temporary artificial records test CSV handling only; never saved as paper data.
        case = next(c for c in self.cases if c['runner']=='openmp')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root/case['figure']/'cases'/case['id']
            run = base/'attempt_test'
            run.mkdir(parents=True)
            with (run/'openmp_raw_latest_test.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=['status','time_sec','log_file'])
                writer.writeheader()
                writer.writerows([{'status':'OK','time_sec':'1','log_file':'a,b'},
                                  {'status':'BUILD_FAILED','time_sec':'NA'}])
            pc.save_json(base/'status.json',dict(status='failed',attempt='attempt_test'))
            pc.aggregate(root,[case])
            with (root/case['figure']/'raw_data.csv').open(newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['log_file'],'a,b')
            self.assertEqual(rows[1]['status'],'BUILD_FAILED')


if __name__ == '__main__':
    unittest.main()
