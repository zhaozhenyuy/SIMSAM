import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

import testmemsam
import trainmemsam
from models.segment_anything_memsam.modeling.mem import Mem
from utils.loss_functions.sam_loss import get_criterion
from utils.release_checkpoints import load_release_checkpoint
from utils.data_us import EchoVideoDataset
from utils.lvef_evaluation import save_camus_lvef
from utils.visualization import _write_image


class ReleaseContractTests(unittest.TestCase):
    def test_test_defaults_match_reference_except_mamba(self):
        args = testmemsam.parse_args(['--load_path', 'model.pth'])
        self.assertFalse(args.reinforce)
        self.assertTrue(args.enable_memory)
        self.assertFalse(args.enable_phase_memory)
        self.assertFalse(args.enable_apfe)
        self.assertFalse(args.enable_es_shape_loss)
        self.assertEqual(args.es_loss_weight, 1.)
        self.assertEqual(args.es_boundary_loss_weight, .2)
        self.assertEqual(args.es_area_loss_weight, .1)

    def test_only_mamba_checkpoint_tensors_are_ignored(self):
        model = nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            state = model.state_dict()
            state['memory.memory_reinforce.pre_conv.0.weight'] = torch.zeros(1)
            torch.save(state, path)
            load_release_checkpoint(model, path, 'cpu')
            state['phase_memory_scale'] = torch.zeros(1)
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_release_checkpoint(model, path, 'cpu')
            del state['phase_memory_scale']
            del state['weight']
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_release_checkpoint(model, path, 'cpu')

    def test_mamba_cannot_be_enabled(self):
        with self.assertRaises(ValueError):
            Mem({'key_dim': 64, 'value_dim': 256, 'hidden_dim': 64, 'reinforce': True})

    def test_basic_hidden_update_is_retained(self):
        memory = Mem({'key_dim': 64, 'value_dim': 256, 'hidden_dim': 64, 'reinforce': False})
        self.assertIsNotNone(memory.value_encoder.hidden_reinforce)
        self.assertIsNotNone(memory.decoder.hidden_update)
        self.assertFalse(hasattr(memory, 'memory_reinforce'))

    def test_original_joint_endpoint_loss(self):
        criterion = get_criterion('SharedGroundedMemSAM', SimpleNamespace(device='cpu'))
        prediction = torch.randn(1, 10, 1, 12, 12)
        target = torch.randint(0, 2, (1, 10, 12, 12)).float()
        loss = trainmemsam.endpoint_weighted_loss(criterion, prediction, target, 1.)
        expected = criterion(prediction[:, [0, -1], 0], target[:, [0, -1]])
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)

    def test_windows_spawn_without_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'videos/test').mkdir(parents=True)
            (root / 'annotations/test').mkdir(parents=True)
            (root / 'class.json').write_text('{"camus": 2}', encoding='utf-8')
            np.save(root / 'videos/test/patient0001_2CH.npy', np.zeros((3, 10, 12, 12), np.uint8))
            np.savez(root / 'annotations/test/patient0001_2CH.npz',
                     fnum_mask={'0': np.ones((12, 12), np.uint8), '9': np.ones((12, 12), np.uint8)},
                     ef=60., edv=100., esv=40., spacing=np.ones(3))
            dataset = EchoVideoDataset(str(root), split='test', frame_length=10)
            loader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context='spawn')
            batch = next(iter(loader))
            self.assertEqual(tuple(batch['image'].shape), (1, 10, 3, 12, 12))

    def test_lvef_four_metrics_and_export(self):
        def volume_from_masks(a2c_ed, a2c_es, **kwargs):
            return float(a2c_ed.sum()), float(a2c_es.sum())

        cases, references = {}, {}
        for index, reference in enumerate((40., 60., 70.), 1):
            es = np.zeros((10, 10), np.uint8)
            es.flat[:int(102 - reference)] = 1
            views = {'ED': np.ones((10, 10), np.uint8), 'ES': es, 'spacing': np.ones(2)}
            name = f'patient{index:04d}'
            cases[name] = {'2CH': views, '4CH': views}
            references[name] = reference
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(compute_ef=True, clinical_output_dir=directory)
            opt = SimpleNamespace(eval_mode='camus', batch_size=1, data_path='fixture', test_split='test')
            save_camus_lvef(cases, references, args, opt, volume_from_masks)
            self.assertTrue((Path(directory) / 'clinical_per_patient.csv').is_file())
            summary = json.loads((Path(directory) / 'clinical_summary.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(summary['bias'], -2.)
            self.assertAlmostEqual(summary['mae'], 2.)

    def test_unicode_visualization_path(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / '\u4e2d\u6587.png'
            _write_image(str(filename), np.zeros((12, 12, 3), np.uint8))
            self.assertTrue(filename.is_file())


if __name__ == '__main__':
    unittest.main()
