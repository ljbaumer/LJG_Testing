from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd

# Define conversion factors for performance units (all in FLOPS)
GIGA_FLOP = 1e9
TERA_FLOP = 1e12
PETA_FLOP = 1e15

@dataclass
class GPUDataclass:
    name: str
    wattage: float  # in watts
    ai_accelerator_price: float  # Chip vendor ASP (base chip cost) in dollars
    other_compute_costs: float  # Additional costs = (total price - ai_accelerator_price) in dollars
    hourly_revenue: Optional[float] = None  # Dollars per hour
    quantized_performance: Optional[Dict[str, Optional[float]]] = None  # Performance metrics in FLOPS
    data_sources: Optional[List[str]] = None  # List of data source URLs

# GPU instances with hardcoded pricing numbers

A100 = GPUDataclass(
    name="A100",
    wattage=1000,
    ai_accelerator_price=5000,         # Chip vendor ASP: $5,000.00
    other_compute_costs=2500,  # Total: $7,500.00 (Non-GPU: $2,500.00; 33.3% of total)
    hourly_revenue=1.79,
    quantized_performance={
        'fp64': 9.7 * TERA_FLOP,
        'tf64': 19.5 * TERA_FLOP,
        'fp32': 19.5 * TERA_FLOP,
        'tf32': 156 * TERA_FLOP,
        'bf16': 312 * TERA_FLOP,
        'fp16': 312 * TERA_FLOP,
        'int8': 624 * TERA_FLOP,
    },
    data_sources=[
        "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",
        "https://www.amazon.com/NVIDIA-Ampere-Graphics-Processor-Accelerator/dp/B08X13X6HF"
    ]
)

H100 = GPUDataclass(
    name="H100",
    wattage=1350,
    ai_accelerator_price=25000,         # Chip vendor ASP: $25,000.00
    other_compute_costs=12500,  # Total: $37,500.00 (Non-GPU: $12,500.00; 33.3% of total)
    hourly_revenue=2.00,
    quantized_performance={
        'fp64': 34 * TERA_FLOP,
        'tf64': 67 * TERA_FLOP,
        'fp32': 67 * TERA_FLOP,
        'tf32': 989 * TERA_FLOP,
        'bf16': 1.979 * PETA_FLOP,
        'fp16': 1.979 * PETA_FLOP,
        'fp8': 3.958 * PETA_FLOP,
        'int8': 3.958 * PETA_FLOP,
    },
    data_sources=[
        "https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet",
        "https://marketplace.uvation.com/nvidia-h100-tensor-core-gpu-80gb-pcie/?srsltid=AfmBOopnOxWaHsLquhBS0IRMkZH81BwNRG9HHfhUJIS2m85sgOXE34n3y8U"
    ]
)

H100_Depreciated = GPUDataclass(
    name="H100_Depreciated",
    wattage=1350,
    ai_accelerator_price=1,         # Fully depreciated
    other_compute_costs=1,  # No additional costs
    hourly_revenue=0.00,    # No revenue generation
    quantized_performance={
        'fp64': 34 * TERA_FLOP,
        'tf64': 67 * TERA_FLOP,
        'fp32': 67 * TERA_FLOP,
        'tf32': 989 * TERA_FLOP,
        'bf16': 1.979 * PETA_FLOP,
        'fp16': 1.979 * PETA_FLOP,
        'fp8': 3.958 * PETA_FLOP,
        'int8': 3.958 * PETA_FLOP,
    },
    data_sources=["Fully depreciated H100 GPU"]
)

H200 = GPUDataclass(
    name="H200",
    wattage=1300,
    ai_accelerator_price=27500,         # Chip vendor ASP: $27,500.00
    other_compute_costs=12500,  # Total: $40,000.00 (Non-GPU: $12,500.00; 31.2% of total)
    hourly_revenue=2.50,
    quantized_performance={
        'fp64': 42.5 * TERA_FLOP,
        'tf64': 83.75 * TERA_FLOP,
        'fp32': 83.75 * TERA_FLOP,
        'tf32': 1236.25 * TERA_FLOP,
        'bf16': 2.47 * PETA_FLOP,
        'fp16': 2.47 * PETA_FLOP,
        'int8': 4.95 * PETA_FLOP,
        'fp8': 4.95 * PETA_FLOP,
    },
    data_sources=["Estimated based on H100 specifications with 25% performance increase"]
)

B100 = GPUDataclass(
    name="B100",
    wattage=1300,
    ai_accelerator_price=30000,         # Chip vendor ASP: $30,000.00
    other_compute_costs=12500,  # Total: $42,500.00 (Non-GPU: $12,500.00; 29.4% of total)
    hourly_revenue=3.00,
    quantized_performance={
        'fp64': None,
        'tf64': 30 * TERA_FLOP,
        'fp32': None,
        'tf32': 0.9 * PETA_FLOP,
        'bf16': 1.8 * PETA_FLOP,
        'fp16': 1.8 * PETA_FLOP,
        'int8': 3.5 * PETA_FLOP,
        'fp8': 3.5 * PETA_FLOP,
        'fp4': 7 * PETA_FLOP,
    },
    data_sources=[
        "https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis",
        "https://www.tomshardware.com/pc-components/gpus/nvidias-next-gen-ai-gpu-revealed-blackwell-b200-gpu-delivers-up-to-20-petaflops-of-compute-and-massive-improvements-over-hopper-h100"
    ]
)


GB200 = GPUDataclass(
    name="GB200",
    wattage=1850,
    ai_accelerator_price=37500,         # Chip vendor ASP: $37,500.00
    other_compute_costs=14936,  # Total: $52,436.00 (Non-GPU: $14,936.00; 28.5% of total)
    hourly_revenue=3.25,
    quantized_performance={
        'fp64': None,
        'tf64': 40 * TERA_FLOP,
        'fp32': None,
        'tf32': 1.12 * PETA_FLOP,
        'bf16': 2.25 * PETA_FLOP,
        'fp16': 2.25 * PETA_FLOP,
        'int8': 4.5 * PETA_FLOP,
        'fp8': 4.5 * PETA_FLOP,
        'fp4': 9 * TERA_FLOP,
    },
    data_sources=[
        "https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis",
        "https://www.tomshardware.com/pc-components/gpus/nvidias-next-gen-ai-gpu-revealed-blackwell-b200-gpu-delivers-up-to-20-petaflops-of-compute-and-massive-improvements-over-hopper-h100"
    ]
)

GB300 = GPUDataclass(
    name="GB300",
    wattage=2080,
    ai_accelerator_price=41500,         # Chip vendor ASP: $41,500.00
    other_compute_costs=15087,  # Total: $56,587.00 (Non-GPU: $15,087.00; 26.7% of total)
    hourly_revenue=3.13,
    quantized_performance={
        'fp64': None,
        'tf64': None,
        'fp32': None,
        'tf32': None,
        'bf16': 3.15 * PETA_FLOP,
        'fp16': 3.15 * PETA_FLOP,
        'int8': 6.3 * PETA_FLOP,
        'fp8': 6.3 * PETA_FLOP,
        'fp4': 12.6 * PETA_FLOP,
    },
    data_sources=["https://semianalysis.com/2025/01/23/openai-stargate-joint-venture-demystified/"]
)

VR200 = GPUDataclass(
    name="VR200",
    wattage=3000,
    ai_accelerator_price=63750,         # Chip vendor ASP: $63,750.00
    other_compute_costs=16822,  # Total: $80,572.00 (Non-GPU: $16,822.00; 20.9% of total)
    hourly_revenue=4.47,
    quantized_performance={
        'fp64': None,
        'tf64': None,
        'fp32': None,
        'tf32': None,
        'bf16': 5.5 * PETA_FLOP,
        'fp16': 5.5 * PETA_FLOP,
        'int8': 11 * PETA_FLOP,
        'fp8': 11 * PETA_FLOP,
        'fp4': 22 * PETA_FLOP,
    },
    data_sources=["https://semianalysis.com/2025/01/23/ope nai-stargate-joint-venture-demystified/"]
)

MI450X = GPUDataclass(
    name="MI450X",
    wattage=2600,  # SemiAnalysis: 1600-2000W TDP range for CDNA-NEXT, then add 800 watts for everything else in the node per GPU
    ai_accelerator_price=int(VR200.ai_accelerator_price * 0.8),  # 20% haircut on Rubin
    other_compute_costs=int(VR200.other_compute_costs * 0.8),    # 20% haircut on Rubin
    hourly_revenue=round(VR200.hourly_revenue * 0.8, 2),
    quantized_performance={
        'fp64': None,
        'tf64': None,
        'fp32': None,
        'tf32': None,
        'bf16': VR200.quantized_performance['bf16'] * 0.9,  # 10% below Rubin performance
        'fp16': VR200.quantized_performance['fp16'] * 0.9,  # 10% below Rubin performance
        'int8': VR200.quantized_performance['int8'] * 0.9,
        'fp8': VR200.quantized_performance['fp8'] * 0.9,
        'fp4': VR200.quantized_performance['fp4'] * 0.9,
    },
    data_sources=["SemiAnalysis MI450X CDNA-NEXT H2 2026: 20% haircut on Rubin pricing, 10% lower performance"]
)

ALL_GPU_LIST = [A100, H100, H100_Depreciated, H200, B100, GB200, GB300, VR200, MI450X]

def gpus_to_dataframe(gpus: List[GPUDataclass]) -> pd.DataFrame:
    data = []
    for gpu in gpus:
        gpu_dict = asdict(gpu)
        # If performance data exists, flatten it into the main dictionary
        if gpu_dict.get('quantized_performance'):
            gpu_dict.update(gpu_dict.pop('quantized_performance'))
        data.append(gpu_dict)
    return pd.DataFrame(data)

# Example usage: list of GPUs (B100 has been removed)
if __name__ == "__main__":
    gpu_df = gpus_to_dataframe(ALL_GPU_LIST)
    print(gpu_df)
