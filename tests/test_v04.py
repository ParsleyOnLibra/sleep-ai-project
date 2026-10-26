import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eeg'))
from transitions import evaluate
from sustained import bouts, evaluate_sustained

class RegressionTests(unittest.TestCase):
    def test_quality_channel_reasons(self):
        from features import quality_flags
        signal=np.tile(np.sin(np.arange(3000))*1e-5,(3,1))
        muscle=np.ones(30)
        self.assertEqual(quality_flags(signal,muscle),'')
        signal[0]=0;signal[1]*=100;signal[2,0]=np.nan
        reason=quality_flags(signal,muscle)
        self.assertIn('EEG Fpz-Cz:flat',reason)
        self.assertIn('EEG Pz-Oz:high_amplitude',reason)
        self.assertIn('EOG horizontal:nonfinite',reason)

    def test_future_reference_cannot_match(self):
        y=np.array([0,0,0,0,1,1]); p=np.eye(5)[[0,0,1,1,0,0]]
        result=evaluate(y,p,np.arange(6)*30,np.ones(6,bool))
        self.assertEqual(result['matched'],0)
        self.assertEqual(result['missed'],1)

    def test_start_and_confirmation_are_separate(self):
        y=np.array([0,1,2,2,3,3]); t=np.arange(6)*30; v=np.ones(6,bool)
        events=bouts(y,t,v,2)
        self.assertEqual(events,[dict(start_s=30.,confirmation_s=150.,end_s=180.)])
        self.assertEqual(evaluate_sustained(y,np.eye(5)[y],t,v,2)['recall'],1)

    def test_gap_qc_and_wake_reset(self):
        y=np.ones(6,int); t=np.arange(6)*30; v=np.ones(6,bool)
        v[2]=False; self.assertEqual(bouts(y,t,v,2),[])
        v[:]=True;t[3:]+=30; self.assertEqual(bouts(y,t,v,2),[])
        t=np.arange(6)*30;y[2]=0;self.assertEqual(bouts(y,t,v,2),[])

    def test_future_does_not_change_confirmation(self):
        y=np.ones(12,int);t=np.arange(12)*30;v=np.ones(12,bool)
        a=bouts(y[:4],t[:4],v[:4],2)[0]
        b=bouts(y,t,v,2)[0]
        self.assertEqual(a['start_s'],b['start_s'])
        self.assertEqual(a['confirmation_s'],b['confirmation_s'])

    def test_premature_confirmation_is_false(self):
        y=np.array([0,0,1,1,1,1,1,1]);pred=np.ones(8,int)
        result=evaluate_sustained(y,np.eye(5)[pred],np.arange(8)*30,np.ones(8,bool),2)
        self.assertEqual(result['matched'],0)

if __name__=='__main__': unittest.main()
