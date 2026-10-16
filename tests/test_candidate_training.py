"""CPU-only integration checks; no model fitting or downloads."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REQUEST_STACK = os.environ.get('ATLAS_CONTAINER_CPU_TESTS') == '1'
HAS_STACK = REQUEST_STACK and all(importlib.util.find_spec(n) is not None for n in ('torch', 'torchvision', 'numpy', 'PIL'))
if REQUEST_STACK and not HAS_STACK:
    raise RuntimeError('Explicit container CPU gate requires the full training stack')


@unittest.skipUnless(HAS_STACK, 'CPU training dependencies unavailable; executed in candidate image')
class CandidateDatasetTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'configs/smoke/candidate-v2/none.json').read_text())

    def test_candidate_resize_precedes_augmentation_and_evaluation_has_no_augmentation(self):
        from PIL import Image
        from augment_atlas.train import ClassificationDataset
        cfg=self.config()
        with tempfile.TemporaryDirectory() as temporary:
            Image.new('L',(43,79),95).save(Path(temporary)/'fixture.png')
            cfg['dataset'].update(adapter='manifest_csv',root=temporary)
            records=[dict(id='fixture',path='fixture.png',group='fixture-group',label=1)]
            observed=[]
            def operation(image,augmentation,identity):
                observed.append((image.size,identity));return image
            with patch.dict(os.environ,{'ATLAS_DATA_ROOT':temporary}),patch('augment_atlas.augment.apply_candidate_augmentation',side_effect=operation):
                tensor,label=ClassificationDataset(cfg,records,epoch=7,training=True)[0]
                side=cfg['input']['size']
                self.assertEqual(observed,[((side,side),[3,7,'fixture'])])
                self.assertEqual(tuple(tensor.shape),(3,side,side));self.assertEqual(label,1)
                ClassificationDataset(cfg,records,epoch=7,training=False)[0]
                self.assertEqual(len(observed),1)

    def test_legacy_resize_still_follows_augmentation(self):
        from PIL import Image
        from augment_atlas.train import ClassificationDataset
        cfg=json.loads((ROOT/'configs/smoke/none.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            Image.new('L',(43,79),95).save(Path(temporary)/'fixture.png')
            cfg['dataset'].update(adapter='manifest_csv',root=temporary)
            records=[dict(id='fixture',path='fixture.png',group='fixture-group',label=0)]
            sizes=[]
            def operation(image,*args):sizes.append(image.size);return image
            with patch.dict(os.environ,{'ATLAS_DATA_ROOT':temporary}),patch('augment_atlas.augment.apply_augmentation',side_effect=operation):
                ClassificationDataset(cfg,records,training=True)[0]
            self.assertEqual(sizes,[(43,79)])

    def test_changed_weight_cache_rejected_before_deserializing(self):
        import torch
        from augment_atlas.train import run
        cfg=self.config();cfg['training']['device']='cpu'
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);(folder/'hub/checkpoints').mkdir(parents=True)
            (folder/'hub/checkpoints/resnet50-11ad3fa6.pth').write_bytes(b'invalid-fixture-weights')
            with patch('torch.hub.get_dir',return_value=str(folder/'hub')),patch('torchvision.models.resnet50'),patch('torch.load') as load:
                with self.assertRaisesRegex(ValueError,'ImageNet cache checksum'):
                    run(cfg,folder/'output')
                load.assert_not_called()


if __name__=='__main__':unittest.main()
