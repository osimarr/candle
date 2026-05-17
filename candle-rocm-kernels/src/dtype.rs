#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelDType {
    U8,
    U32,
    I16,
    I32,
    I64,
    BF16,
    F16,
    F32,
    F64,
    F8E4M3,
    F6E2M3,
    F6E3M2,
    F4,
    F8E8M0,
}

impl KernelDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F8E4M3 => "f8e4m3",
            Self::F6E2M3 => "f6e2m3",
            Self::F6E3M2 => "f6e3m2",
            Self::F4 => "f4",
            Self::F8E8M0 => "f8e8m0",
        }
    }

    pub fn bits_per_element(self) -> usize {
        match self {
            Self::U8 => 8,
            Self::U32 => 32,
            Self::I16 => 16,
            Self::I32 => 32,
            Self::I64 => 64,
            Self::BF16 => 16,
            Self::F16 => 16,
            Self::F32 => 32,
            Self::F64 => 64,
            Self::F8E4M3 => 8,
            Self::F6E2M3 => 6,
            Self::F6E3M2 => 6,
            Self::F4 => 4,
            Self::F8E8M0 => 8,
        }
    }

    pub fn size_in_bytes(self) -> Option<usize> {
        let bits = self.bits_per_element();
        bits.is_multiple_of(8).then_some(bits / 8)
    }

    pub fn storage_size_in_bytes(self, elem_count: usize) -> usize {
        (self.bits_per_element() * elem_count).div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::KernelDType;

    #[test]
    fn packed_dtype_storage_sizes_round_up_to_bytes() {
        assert_eq!(KernelDType::F4.storage_size_in_bytes(3), 2);
        assert_eq!(KernelDType::F6E2M3.storage_size_in_bytes(3), 3);
        assert_eq!(KernelDType::F32.storage_size_in_bytes(3), 12);
    }
}
