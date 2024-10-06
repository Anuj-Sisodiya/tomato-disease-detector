# tests/test_load_data.py

import unittest
from src.data.load_data import get_transforms, load_datasets

class TestLoadData(unittest.TestCase):
    def test_get_transforms(self):
        transform = get_transforms()
        self.assertIsNotNone(transform)

    def test_load_datasets(self):
        train_dir = '/path/to/train'
        valid_dir = '/path/to/valid'
        transform = get_transforms()
        try:
            trainset, trainloader, testset, testloader = load_datasets(train_dir, valid_dir, transform, batch_size=32)
            self.assertTrue(len(trainset) > 0)
            self.assertTrue(len(testset) > 0)
        except Exception as e:
            self.fail(f'load_datasets raised an exception {e}')

if __name__ == '__main__':
    unittest.main()
