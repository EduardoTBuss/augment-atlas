"""The frozen metric decision is shared across arms/seeds and changes recipe ID."""
import copy
from pathlib import Path
import unittest

from augment_atlas.cli import recipe_identity
from augment_atlas.contracts import canonical_hash,load_config

ROOT=Path(__file__).resolve().parents[1]


class EvaluationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(ROOT/'configs/smoke/synthetic.json')

    def test_legacy_identity_is_byte_for_byte_compatible(self):
        cfg=self.cfg
        before=copy.deepcopy(cfg)
        legacy=canonical_hash(dict(recipe=cfg['recipe'],model=cfg['model'],input=cfg['input'],
            training={k:v for k,v in cfg['training'].items() if k!='seed'}))
        self.assertEqual(recipe_identity(cfg),legacy)
        self.assertEqual(cfg,before)

    def test_every_frozen_evaluation_decision_changes_identity(self):
        cfg=self.cfg
        evaluation=dict(schema_version=1,task='binary',classes=[0,1],score_kind='softmax_probability',
                        frozen=True,positive_class=1,threshold=0.5,comparison='ge')
        cfg['evaluation']=evaluation
        baseline=recipe_identity(cfg)
        without=copy.deepcopy(cfg);without.pop('evaluation')
        self.assertNotEqual(recipe_identity(without),baseline)
        for field,value in [('positive_class',0),('threshold',0.6),('comparison','gt'),('classes',[1,0])]:
            changed=copy.deepcopy(cfg);changed['evaluation'][field]=value
            with self.subTest(field=field):self.assertNotEqual(recipe_identity(changed),baseline)
        changed=copy.deepcopy(cfg);changed['training']['seed']+=1
        self.assertEqual(recipe_identity(changed),baseline)
        changed=copy.deepcopy(cfg);changed['augmentation']={'arm':'basic','ops':[{'op':'basic'}]}
        self.assertEqual(recipe_identity(changed),baseline)


if __name__=='__main__':unittest.main()
