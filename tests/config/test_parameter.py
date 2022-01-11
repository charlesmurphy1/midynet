import unittest

from midynet import config


class TestParameter(unittest.TestCase):
    def setUp(self):
        self.name = "x"
        self.int_value = 10
        self.float_value = 10.0
        self.list_int_value = [1, 2, 3, 4, 5]

    def test_datatype(self):
        p = config.Parameter(name=self.name, value=self.int_value)
        self.assertTrue(p.datatype == int)

        p = config.Parameter(name=self.name, value=self.float_value)
        self.assertTrue(p.datatype == float)

        p = config.Parameter(name=self.name, value=self.list_int_value)
        self.assertTrue(p.datatype == int)

    def test_getitem(self):
        p = config.Parameter(name=self.name, value=self.int_value)
        with self.assertRaises(LookupError):
            p[0]

        p = config.Parameter(name=self.name, value=self.list_int_value)
        self.assertEqual(p[0], self.list_int_value[0])

    def test_set_value(self):
        p = config.Parameter(name=self.name, value=self.int_value)
        p.set_value(2.5)
        self.assertTrue(p.datatype, float)
        self.assertEqual(p.value, 2.5)

    def test_is_sequenced(self):
        p = config.Parameter(name=self.name, value=self.int_value)
        self.assertFalse(p.is_sequenced())

        p = config.Parameter(name=self.name, value=self.list_int_value)
        self.assertTrue(p.is_sequenced())

    def test_is_unique(self):
        p = config.Parameter(name=self.name, value=self.int_value, unique=False)
        self.assertFalse(p.is_unique())

        p = config.Parameter(name=self.name, value=self.int_value, unique=True)
        self.assertTrue(p.is_unique())

    def test_infer_type(self):
        p = config.Parameter(name=self.name, value=self.int_value)

        self.assertEqual(p.infer_type(1), int)
        self.assertEqual(p.infer_type([1]), int)
        self.assertEqual(p.infer_type([[1]]), int)
        self.assertEqual(p.infer_type([(1,)]), int)

        p = config.Parameter(
            name=self.name, value=self.int_value, force_non_sequence=True
        )
        self.assertEqual(list, p.infer_type([1]))

    def test_generate_sequence(self):
        p = config.Parameter(name=self.name, value=self.list_int_value)
        for pp, x in zip(p.generate_sequence(), self.list_int_value):
            self.assertEqual(pp, x)

    def test_add_value_to_nonsequenced_parameter(self):
        p = config.Parameter(name="x", value=1)
        self.assertFalse(p.is_sequenced())

        p.add_value(2)
        self.assertEqual(p.value, {1, 2})
        self.assertTrue(p.is_sequenced())

    def test_add_value_to_sequenced_parameter(self):
        p = config.Parameter(name="x", value={1.1, 2})
        self.assertTrue(p.is_sequenced())

        p.add_value(2)
        self.assertEqual(p.value, {1.1, 2})

        p.add_value(3)
        self.assertEqual(p.value, {1.1, 2, 3})

    def test_add_values_to_nonsequenced_parameter(self):
        p = config.Parameter(name="x", value=1.1)
        self.assertFalse(p.is_sequenced())

        p.add_values({2, 3})
        self.assertEqual(p.value, {1.1, 2, 3})
        self.assertTrue(p.is_sequenced())

    def test_add_values_to_sequenced_parameter(self):
        p = config.Parameter(name="x", value={1.1, 2})
        self.assertTrue(p.is_sequenced())

        p.add_values({2, 3})
        self.assertEqual(p.value, {1.1, 2, 3})

    # def infer_type(self, value: typing.Any):
    #
    # def generate_sequence(self):
