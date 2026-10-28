"""Verify fifth-run rendering and warmup-only runs without changing older curves."""
import csv
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if find_spec('matplotlib') is None:
    raise unittest.SkipTest('Plotting tests require matplotlib')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/reward_plots'))
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from plot_orm_comparison import COMPONENTS
from plot_reward_grids import main


class FifthRunGridTest(unittest.TestCase):
    def render(self, skip=False):
        labels=['Dense ORM','Sparse ORM','PPO + Dense ORM','PPO + Turn rewards','CISPO + Turn critic (K=4)']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'summary.json').write_text(json.dumps({'mean_only_runs':labels[3:]}))
            rows=[dict(run=run,step=i,component=component,rollouts=512,
                       mean=i/100,max_at_8=.5,nonzero_advantage=.2)
                  for run in labels for i in range(4 if run==labels[-1] else 16)
                  for component,_ in COMPONENTS]
            with (root/'metrics.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            curves=[];legends=[]
            def plot(_self,x,y,**kwargs):
                curves.append((list(x),list(y),kwargs.get('color')))
                return []
            def legend(_self,**kwargs):
                legends.extend(handle.get_label() for handle in kwargs['handles'])
            args=[str(root),'--output-dir',str(root/'out'),'--combined','--smooth-only',
                  '--dynamic-ylim','--alignment','trailing','--window','16']
            if skip:
                args+=['--skip-first-steps',labels[3],'12','--skip-first-steps',labels[4],'12',
                       '--rebase-skipped-steps','--allow-empty-after-skip']
            with patch.object(Axes,'plot',plot),patch.object(Figure,'legend',legend),patch.object(Figure,'savefig'):
                main(args)
            return curves,legends,json.loads((root/'out/plot_options.json').read_text())

    def test_fifth_run_uses_mean_panels_only(self):
        curves,labels,options=self.render()
        self.assertEqual(sum(c[2]=='#c44e52' for c in curves),11)
        self.assertEqual(len(labels),5)
        self.assertEqual(options['omitted_runs'],[])

    def test_warmup_only_run_has_label_without_curve(self):
        curves,labels,options=self.render(skip=True)
        self.assertFalse(any(c[2]=='#c44e52' for c in curves))
        self.assertEqual(options['omitted_runs'],['CISPO + Turn critic (K=4)'])
        self.assertIn('CISPO + Turn critic (K=4) (no post-warmup data)',labels)
        purple=next(c for c in curves if c[2]=='#9361b0')
        self.assertEqual(purple[0],[0,1,2,3])
        self.assertAlmostEqual(purple[1][0],.12)  # No values from excluded warmup.
        self.assertAlmostEqual(purple[1][-1],.135)
        blue=next(c for c in curves if c[2]=='#1673b8')
        self.assertEqual(blue[0],list(range(16)))
        self.assertAlmostEqual(blue[1][-1],.075)


if __name__=='__main__':
    unittest.main()
