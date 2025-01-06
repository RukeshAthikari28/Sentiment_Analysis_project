import unittest
class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('rukesh'.upper(), 'RUKESH')

    def test_isupper(self):
        self.assertTrue('RUKESH'.isupper())
        self.assertFalse('rukesh'.isupper())

    def test_split(self):
        string_input = 'optimal XAI'
        self.assertEqual(string_input.split(), ['optimal', 'XAI'])
        # check that string_input.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            string_input.split(2)
if __name__ == '__main__':
    unittest.main()
