"""Unit tests for TensorDescriptor specialization key generation and parsing.

Tests the round-trip encoding and decoding of host-side TensorDescriptor
specialization keys, which encode max power-of-2 divisibility for shape and
stride dimensions to enable bounded specialization.
"""

from triton.backends.intel.compiler import XPUBackend


class TestTensorDescSpecialization:
    """Test XPUBackend.get_tensordesc_specialization and parse_attr."""

    def test_shape_divisibility_encoding(self):
        """Shape dimensions encode max power-of-2 divisibility."""

        # Mock tensor descriptor with various shape divisibilities
        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # shape=[128, 64, 17] → S0D128S1D64S2D1
        arg = MockArg(shape=[128, 64, 17], strides=[64, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        assert "S0D128" in key, f"Expected S0D128 in {key}"
        assert "S1D64" in key, f"Expected S1D64 in {key}"
        assert "S2D1" in key, f"Expected S2D1 in {key}"

    def test_stride_exact_values_rank3(self):
        """Stride exact values encoded for rank-3+ descriptors."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # Rank-3: encode non-last strides as exact values
        # strides=[256, 16, 1] → T256,16 (last stride omitted)
        arg = MockArg(shape=[32, 32, 32], strides=[256, 16, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        assert "T256,16" in key, f"Expected T256,16 in {key}"

    def test_stride_not_encoded_rank2(self):
        """Stride values not encoded for rank-2 descriptors."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # Rank-2: no stride encoding
        arg = MockArg(shape=[128, 64], strides=[64, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        assert "T" not in key, f"Rank-2 should not encode strides, got {key}"

    def test_padding_encoding(self):
        """NaN padding encodes as 'N' prefix."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # With NaN padding
        arg = MockArg(shape=[128, 64], strides=[64, 1], padding="nan")
        key = XPUBackend.get_tensordesc_specialization(arg)
        assert key.startswith("N"), f"Expected 'N' prefix for NaN padding, got {key}"

        # Without padding
        arg = MockArg(shape=[128, 64], strides=[64, 1], padding=None)
        key = XPUBackend.get_tensordesc_specialization(arg)
        assert not key.startswith("N"), f"Should not have 'N' prefix without padding, got {key}"

    def test_parse_attr_shape_divisibility(self):
        """parse_attr correctly decodes shape divisibility."""
        key = "S0D128S1D64S2D1"
        attrs = XPUBackend.parse_attr(key)

        # Convert to dict for easier lookup
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        assert "tt.shape.0.divisibility" in attr_dict
        assert attr_dict["tt.shape.0.divisibility"] == 128
        assert "tt.shape.1.divisibility" in attr_dict
        assert attr_dict["tt.shape.1.divisibility"] == 64
        assert "tt.shape.2.divisibility" in attr_dict
        assert attr_dict["tt.shape.2.divisibility"] == 1

    def test_parse_attr_stride_exact_values(self):
        """parse_attr correctly decodes stride exact values."""
        key = "T256,16"
        attrs = XPUBackend.parse_attr(key)

        attr_dict = {attr[0]: attr[1] for attr in attrs}

        assert "tt.stride.0" in attr_dict
        assert attr_dict["tt.stride.0"] == 256
        assert "tt.stride.1" in attr_dict
        assert attr_dict["tt.stride.1"] == 16

    def test_parse_attr_padding(self):
        """parse_attr correctly decodes padding flag."""
        key = "N"
        attrs = XPUBackend.parse_attr(key)

        attr_dict = {attr[0]: attr[1] for attr in attrs}

        assert "tt.padding" in attr_dict
        assert attr_dict["tt.padding"] == 1

    def test_roundtrip_full_example(self):
        """Full round-trip: encode → key → parse → attrs."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # Rank-3 descriptor with NaN padding
        arg = MockArg(shape=[256, 128, 16], strides=[2048, 16, 1], padding="nan")

        # Encode
        key = XPUBackend.get_tensordesc_specialization(arg)

        # Parse
        attrs = XPUBackend.parse_attr(key)
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        # Verify all expected attributes
        assert attr_dict["tt.padding"] == 1
        assert attr_dict["tt.shape.0.divisibility"] == 256
        assert attr_dict["tt.shape.1.divisibility"] == 128
        assert attr_dict["tt.shape.2.divisibility"] == 16
        assert attr_dict["tt.stride.0"] == 2048
        assert attr_dict["tt.stride.1"] == 16

    def test_divisibility_power_of_2(self):
        """Shape divisibility values are always powers of 2."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # Test various non-power-of-2 values
        arg = MockArg(shape=[96, 100, 1023], strides=[10000, 1023, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        attrs = XPUBackend.parse_attr(key)
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        # Shape divisibility is power-of-2
        # 96 = 32 * 3 → max divisor = 32
        assert attr_dict["tt.shape.0.divisibility"] == 32
        # 100 = 4 * 25 → max divisor = 4
        assert attr_dict["tt.shape.1.divisibility"] == 4
        # 1023 = odd → max divisor = 1
        assert attr_dict["tt.shape.2.divisibility"] == 1

        # Stride values are exact (not divisibility)
        # 10000 encoded as exact value
        assert attr_dict["tt.stride.0"] == 10000
        # 1023 encoded as exact value
        assert attr_dict["tt.stride.1"] == 1023

    def test_zero_value_divisibility(self):
        """Zero values have divisibility of 1."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        arg = MockArg(shape=[0, 128], strides=[0, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        attrs = XPUBackend.parse_attr(key)
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        # Zero should get divisibility 1
        assert attr_dict["tt.shape.0.divisibility"] == 1
