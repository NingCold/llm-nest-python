from enum import Enum


class QuantType(str, Enum):
    F32 = "F32"
    F16 = "F16"
    Q8_0 = "Q8_0"
    Q8_1 = "Q8_1"
    Q6_K = "Q6_K"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q5_K_M = "Q5_K_M"
    Q5_K_S = "Q5_K_S"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q4_K_M = "Q4_K_M"
    Q4_K_S = "Q4_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_S = "Q3_K_S"
    Q2_K = "Q2_K"
    IQ4_XS = "IQ4_XS"
    IQ3_XXS = "IQ3_XXS"
    IQ2_XXS = "IQ2_XXS"
    UNKNOWN = "UNKNOWN"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
