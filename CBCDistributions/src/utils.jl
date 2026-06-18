const JULIAN_YEAR_SEC = 365.25 * 24 * 3600.0

year_to_second(yr::Real) = Float64(yr) * JULIAN_YEAR_SEC
second_to_year(sec::Real) = Float64(sec) / JULIAN_YEAR_SEC
