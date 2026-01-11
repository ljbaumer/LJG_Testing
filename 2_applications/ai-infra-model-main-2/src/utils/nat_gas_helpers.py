# Pure functions for natural gas calculations.

def convert_mmbtu_to_dollars(mmbtu: float, dollars_per_mmbtu: float) -> float:
    """
    Convert MMBTU to dollars.
    
    Args:
        mmbtu: Amount in MMBTU (must be positive)
        dollars_per_mmbtu: Cost per MMBTU in dollars (must be positive)
        
    Returns:
        Cost in dollars
    """
    if mmbtu <= 0:
        raise ValueError("MMBTU must be positive")
    if dollars_per_mmbtu <= 0:
        raise ValueError("Dollars per MMBTU must be positive")

    return mmbtu * dollars_per_mmbtu

def convert_mmbtu_to_kwh(mmbtu: float, heat_rate: float) -> float:
    """
    Convert MMBTU to kWh using heat rate.
    
    Args:
        mmbtu: Amount in MMBTU (must be positive)
        heat_rate: Heat rate in BTU/kWh (must be positive)
        
    Returns:
        Energy in kWh
    """
    if mmbtu <= 0:
        raise ValueError("MMBTU must be positive")
    if heat_rate <= 0:
        raise ValueError("Heat rate must be positive")

    return (mmbtu * 1e6) / heat_rate  # Convert MMBTU to BTU then to kWh

def convert_kwh_to_mmbtu(kwh: float, heat_rate: float) -> float:
    """
    Convert kWh to MMBTU using heat rate.
    
    Args:
        kwh: Amount in kWh (must be positive)
        heat_rate: Heat rate in BTU/kWh (must be positive)
        
    Returns:
        Energy in MMBTU
    """
    if kwh <= 0:
        raise ValueError("kWh must be positive")
    if heat_rate <= 0:
        raise ValueError("Heat rate must be positive")

    return (kwh * heat_rate) / 1e6  # Convert kWh to BTU then to MMBTU

def calculate_natural_gas_cost_for_datacenter(
    power_watts: float,
    hours: float,
    heat_rate: float,
    dollars_per_mmbtu: float
) -> float:
    """
    Calculate natural gas cost for a given power consumption.
    
    Args:
        power_watts: Power consumption in watts (must be positive)
        hours: Operating hours (must be positive)
        heat_rate: Heat rate in BTU/kWh (must be positive)
        dollars_per_mmbtu: Cost per MMBTU in dollars (must be positive)
        
    Returns:
        Total cost in dollars
    """
    if power_watts <= 0:
        raise ValueError("Power must be positive")
    if hours <= 0:
        raise ValueError("Hours must be positive")
    if heat_rate <= 0:
        raise ValueError("Heat rate must be positive")
    if dollars_per_mmbtu <= 0:
        raise ValueError("Dollars per MMBTU must be positive")

    # Convert power to kWh
    kwh = (power_watts * hours) / 1000

    # Convert kWh to MMBTU
    mmbtu = convert_kwh_to_mmbtu(kwh, heat_rate)

    # Convert MMBTU to dollars
    return convert_mmbtu_to_dollars(mmbtu, dollars_per_mmbtu)
