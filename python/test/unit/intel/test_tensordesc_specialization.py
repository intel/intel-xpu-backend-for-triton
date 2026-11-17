"""Unit tests for TensorDescriptor specialization key generation and parsing.

Tests the round-trip encoding and decoding of host-side TensorDescriptor
specialization keys, which encode max power-of-2 divisibility (capped at 4)
for shape dimensions to enable bounded specialization.
"""

from triton.backends.intel.compiler import XPUBackend


class TestTensorDescSpecialization:
    """Test XPUBackend.get_tensordesc_specialization and parse_attr."""

    def test_shape_divisibility_encoding(self):
        """Shape dimensions encode max power-of-2 divisibility, capped at 4."""

        # Mock tensor descriptor with various shape divisibilities
        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # shape=[128, 64, 17] → S0D4S1D4S2D1 (128 and 64 capped at 4)
        arg = MockArg(shape=[128, 64, 17], strides=[64, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        assert "S0D4" in key, f"Expected S0D4 in {key}"
        assert "S1D4" in key, f"Expected S1D4 in {key}"
        assert "S2D1" in key, f"Expected S2D1 in {key}"

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
        # Test parsing with capped values (max 4)
        key = "S0D4S1D2S2D1"
        attrs = XPUBackend.parse_attr(key)

        # Convert to dict for easier lookup
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        assert "tt.shape.0.divisibility" in attr_dict
        assert attr_dict["tt.shape.0.divisibility"] == 4
        assert "tt.shape.1.divisibility" in attr_dict
        assert attr_dict["tt.shape.1.divisibility"] == 2
        assert "tt.shape.2.divisibility" in attr_dict
        assert attr_dict["tt.shape.2.divisibility"] == 1

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
        # shape=[256, 128, 16] all capped at 4
        arg = MockArg(shape=[256, 128, 16], strides=[2048, 16, 1], padding="nan")

        # Encode
        key = XPUBackend.get_tensordesc_specialization(arg)

        # Parse
        attrs = XPUBackend.parse_attr(key)
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        # Verify all expected attributes (all capped at 4)
        assert attr_dict["tt.padding"] == 1
        assert attr_dict["tt.shape.0.divisibility"] == 4
        assert attr_dict["tt.shape.1.divisibility"] == 4
        assert attr_dict["tt.shape.2.divisibility"] == 4

    def test_divisibility_power_of_2(self):
        """Shape divisibility values are always powers of 2, capped at 4."""

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

        # Shape divisibility is power-of-2, capped at 4
        # 96 = 32 * 3 → max divisor = 32, capped at 4
        assert attr_dict["tt.shape.0.divisibility"] == 4
        # 100 = 4 * 25 → max divisor = 4
        assert attr_dict["tt.shape.1.divisibility"] == 4
        # 1023 = odd → max divisor = 1
        assert attr_dict["tt.shape.2.divisibility"] == 1

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

    def test_divisibility_cap_at_4(self):
        """Divisibility is capped at 4 for MaterializeBlockPointer alignment."""

        class MockArg:

            def __init__(self, shape, strides, padding=None):
                self.shape = shape
                self.strides = strides
                self.padding = padding

        # Large power-of-2 values should be capped at 4
        arg = MockArg(shape=[1024, 2048, 512], strides=[1048576, 512, 1])
        key = XPUBackend.get_tensordesc_specialization(arg)
        attrs = XPUBackend.parse_attr(key)
        attr_dict = {attr[0]: attr[1] for attr in attrs}

        # All should be capped at 4 (max requirement for fp8)
        assert attr_dict["tt.shape.0.divisibility"] == 4
        assert attr_dict["tt.shape.1.divisibility"] == 4
        assert attr_dict["tt.shape.2.divisibility"] == 4
