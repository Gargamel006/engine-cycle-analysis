import math


BASE_INPUTS = {
    "gamma": 1.4,
    "R": 287.0,
    "h_PR": 42_900_000.0,
}


TURBOFAN_DEFAULTS = {
    **BASE_INPUTS,
    "H": 10668.0,
    "M0": 0.785,
    "alpha": 11.0,
    "pi_f": 1.4,
    "pi_c": 28.6,
    "Tt4": 1923.0,
}


TURBOJET_DEFAULTS = {
    **BASE_INPUTS,
    "H": 10668.0,
    "M0": 0.785,
    "pi_c": 12.0,
    "Tt4": 1600.0,
}


def standard_atmosphere(height_m, gamma=1.4, R=287.0):
    """Return a simplified ISA atmosphere state up to the lower stratosphere."""
    g0 = 9.80665
    sea_level_temperature = 288.15
    sea_level_pressure = 101325.0
    lapse_rate = 0.0065
    tropopause_height = 11000.0
    tropopause_temperature = 216.65

    if height_m <= tropopause_height:
        T0 = sea_level_temperature - lapse_rate * height_m
        P0 = sea_level_pressure * (T0 / sea_level_temperature) ** (g0 / (lapse_rate * R))
    else:
        P11 = sea_level_pressure * (
            tropopause_temperature / sea_level_temperature
        ) ** (g0 / (lapse_rate * R))
        T0 = tropopause_temperature
        P0 = P11 * math.exp(-g0 * (height_m - tropopause_height) / (R * T0))

    rho0 = P0 / (R * T0)
    a0 = math.sqrt(gamma * R * T0)

    return {
        "T0": T0,
        "P0": P0,
        "rho0": rho0,
        "a0": a0,
    }


def _validate_positive(name, value):
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0。")


def calculate_ideal_turbofan(inputs):
    """Calculate an ideal separate-flow turbofan cycle."""
    gamma = inputs["gamma"]
    R = inputs["R"]
    h_PR = inputs["h_PR"]
    H = inputs["H"]
    M0 = inputs["M0"]
    alpha = inputs["alpha"]
    pi_f = inputs["pi_f"]
    pi_c = inputs["pi_c"]
    Tt4 = inputs["Tt4"]

    for name, value in {
        "gamma": gamma,
        "R": R,
        "h_PR": h_PR,
        "M0": M0,
        "pi_f": pi_f,
        "pi_c": pi_c,
        "Tt4": Tt4,
    }.items():
        _validate_positive(name, value)
    if alpha < 0:
        raise ValueError("涵道比 alpha 不能为负。")
    if gamma <= 1:
        raise ValueError("比热比 gamma 必须大于 1。")

    cp = (gamma * R) / (gamma - 1)
    atmosphere = standard_atmosphere(H, gamma=gamma, R=R)
    T0 = atmosphere["T0"]
    P0 = atmosphere["P0"]
    rho0 = atmosphere["rho0"]
    a0 = atmosphere["a0"]
    u0 = M0 * a0

    tau_r = 1 + ((gamma - 1) / 2) * M0**2
    Tt0 = T0 * tau_r
    tau_lambda = Tt4 / T0
    tau_f = pi_f ** ((gamma - 1) / gamma)
    tau_c = pi_c ** ((gamma - 1) / gamma)
    tau_t = 1 - (tau_r / tau_lambda) * ((tau_c - 1) + alpha * (tau_f - 1))

    core_expansion = tau_r * tau_c * tau_t
    if tau_t <= 0 or core_expansion <= 1:
        raise ValueError("涡轮提取能量过多，核心流无法继续膨胀产生推力。")

    fan_expansion = tau_r * tau_f
    if fan_expansion <= 1:
        raise ValueError("风扇出口无法形成有效外涵喷流。")

    u9_a0 = math.sqrt((2 / (gamma - 1)) * tau_lambda * tau_t * (1 - 1 / core_expansion))
    u19_a0 = math.sqrt((2 / (gamma - 1)) * (fan_expansion - 1))
    u9 = u9_a0 * a0
    u19 = u19_a0 * a0

    spec_thrust = ((u9 - u0) + alpha * (u19 - u0)) / (1 + alpha)
    if spec_thrust <= 0:
        raise ValueError("单位推力小于等于 0，此参数组合不适合作为巡航设计点。")

    f = (cp * T0 / h_PR) * (tau_lambda - tau_r * tau_c)
    if f <= 0:
        raise ValueError("燃油空气比小于等于 0，请提高涡轮前总温或降低压气机压缩比。")

    sfc = f / ((1 + alpha) * spec_thrust)
    kinetic_power = (u9**2 - u0**2) + alpha * (u19**2 - u0**2)
    propulsive_power = 2 * u0 * ((u9 - u0) + alpha * (u19 - u0))
    eta_p = propulsive_power / kinetic_power if kinetic_power > 0 else float("nan")
    eta_th = kinetic_power / (2 * f * h_PR)
    eta_o = eta_th * eta_p

    core_thrust = u9 - u0
    bypass_thrust = alpha * (u19 - u0)
    thrust_ratio = bypass_thrust / core_thrust if core_thrust > 0 else float("inf")

    return {
        "T0": T0,
        "P0": P0,
        "rho0": rho0,
        "Tt0": Tt0,
        "a0": a0,
        "u0": u0,
        "tau_r": tau_r,
        "tau_lambda": tau_lambda,
        "tau_f": tau_f,
        "tau_c": tau_c,
        "tau_t": tau_t,
        "u9": u9,
        "u19": u19,
        "u9_u0": u9 / u0 if u0 > 0 else float("nan"),
        "u19_u0": u19 / u0 if u0 > 0 else float("nan"),
        "F_m0": spec_thrust,
        "f": f,
        "SFC": sfc * 1e6,
        "eta_th": eta_th * 100,
        "eta_p": eta_p * 100,
        "eta_o": eta_o * 100,
        "thrust_ratio": thrust_ratio,
    }


def calculate_ideal_turbojet(inputs):
    """Calculate an ideal turbojet cycle."""
    gamma = inputs["gamma"]
    R = inputs["R"]
    h_PR = inputs["h_PR"]
    H = inputs["H"]
    M0 = inputs["M0"]
    pi_c = inputs["pi_c"]
    Tt4 = inputs["Tt4"]

    for name, value in {
        "gamma": gamma,
        "R": R,
        "h_PR": h_PR,
        "M0": M0,
        "pi_c": pi_c,
        "Tt4": Tt4,
    }.items():
        _validate_positive(name, value)
    if gamma <= 1:
        raise ValueError("比热比 gamma 必须大于 1。")

    cp = (gamma * R) / (gamma - 1)
    atmosphere = standard_atmosphere(H, gamma=gamma, R=R)
    T0 = atmosphere["T0"]
    P0 = atmosphere["P0"]
    rho0 = atmosphere["rho0"]
    a0 = atmosphere["a0"]
    u0 = M0 * a0

    tau_r = 1 + ((gamma - 1) / 2) * M0**2
    Tt0 = T0 * tau_r
    tau_lambda = Tt4 / T0
    tau_c = pi_c ** ((gamma - 1) / gamma)
    tau_t = 1 - (tau_r / tau_lambda) * (tau_c - 1)

    expansion = tau_r * tau_c * tau_t
    if tau_t <= 0 or expansion <= 1:
        raise ValueError("涡轮出口能量不足，喷管无法继续膨胀产生推力。")

    u9_a0 = math.sqrt((2 / (gamma - 1)) * tau_lambda * tau_t * (1 - 1 / expansion))
    u9 = u9_a0 * a0
    spec_thrust = u9 - u0
    if spec_thrust <= 0:
        raise ValueError("单位推力小于等于 0，此参数组合不适合作为巡航设计点。")

    f = (cp * T0 / h_PR) * (tau_lambda - tau_r * tau_c)
    if f <= 0:
        raise ValueError("燃油空气比小于等于 0，请提高涡轮前总温或降低压气机压缩比。")

    sfc = f / spec_thrust
    kinetic_power = u9**2 - u0**2
    eta_p = (2 * u0 * (u9 - u0)) / kinetic_power if kinetic_power > 0 else float("nan")
    eta_th = kinetic_power / (2 * f * h_PR)
    eta_o = eta_th * eta_p

    return {
        "T0": T0,
        "P0": P0,
        "rho0": rho0,
        "Tt0": Tt0,
        "a0": a0,
        "u0": u0,
        "tau_r": tau_r,
        "tau_lambda": tau_lambda,
        "tau_c": tau_c,
        "tau_t": tau_t,
        "u9": u9,
        "u9_u0": u9 / u0 if u0 > 0 else float("nan"),
        "F_m0": spec_thrust,
        "f": f,
        "SFC": sfc * 1e6,
        "eta_th": eta_th * 100,
        "eta_p": eta_p * 100,
        "eta_o": eta_o * 100,
    }


def calculate_engine(engine_type, inputs):
    if engine_type == "turbofan":
        return calculate_ideal_turbofan(inputs)
    if engine_type == "turbojet":
        return calculate_ideal_turbojet(inputs)
    raise ValueError(f"未知发动机类型: {engine_type}")


if __name__ == "__main__":
    for engine_type, inputs in {
        "turbofan": TURBOFAN_DEFAULTS,
        "turbojet": TURBOJET_DEFAULTS,
    }.items():
        print(f"\n--- {engine_type} ideal cycle ---")
        for key, value in calculate_engine(engine_type, inputs).items():
            print(f"{key}: {value:.6g}")
