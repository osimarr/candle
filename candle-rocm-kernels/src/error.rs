use crate::KernelDType;

pub type Result<T> = std::result::Result<T, RocmError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RocmError {
    BufferOutOfBounds {
        buffer_bytes: usize,
        offset: usize,
        requested: usize,
    },
    DeviceMismatch {
        expected: usize,
        got: usize,
        op: &'static str,
    },
    InvalidAllocationSize {
        bytes: usize,
    },
    MutexPoisoned(&'static str),
    NotImplemented(&'static str),
    Runtime(String),
    UnsupportedDType {
        dtype: KernelDType,
        op: &'static str,
    },
}

impl std::fmt::Display for RocmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferOutOfBounds {
                buffer_bytes,
                offset,
                requested,
            } => write!(
                f,
                "buffer out of bounds: buffer has {buffer_bytes} bytes, offset {offset}, requested {requested} bytes"
            ),
            Self::DeviceMismatch { expected, got, op } => write!(
                f,
                "device mismatch for {op}: expected ROCm device {expected}, got {got}"
            ),
            Self::InvalidAllocationSize { bytes } => {
                write!(f, "invalid ROCm allocation size {bytes} bytes")
            }
            Self::MutexPoisoned(name) => write!(f, "mutex poisoned while accessing {name}"),
            Self::NotImplemented(op) => write!(f, "ROCm operation {op} is not implemented"),
            Self::Runtime(msg) => write!(f, "ROCm runtime error: {msg}"),
            Self::UnsupportedDType { dtype, op } => {
                write!(f, "unsupported dtype {dtype:?} for ROCm operation {op}")
            }
        }
    }
}

impl std::error::Error for RocmError {}
