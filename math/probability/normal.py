#!/usr/bin/env python3

"""class Normal that represents a normal distribution"""


class Normal:
    """defining the class"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """initiating the constructor"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            x = 0
            for i in range(len(data)):
                x += (data[i] - self.mean) ** 2
            self.stddev = float((x / len(data)) ** 0.5)

    def z_score(self, x):
        """defining the function"""
        return (x-self.mean)/self.stddev

    def x_value(self, z):
        """defining the function"""
        return z*self.stddev + self.mean

    def pdf(self, x):
        """defining the func"""
        PI = 3.1415926536
        E = 2.7182818285
        coefficient = 1 / (self.stddev * (2 * PI) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        pdf = coefficient * (E ** exponent)
        return pdf

    def cdf(self, x):
        PI = 3.1415926536
        value = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = (2 / PI ** 0.5) * (value - (value ** 3)
                                 / 3 + (value ** 5) / 10 - (value ** 7) / 42 +
                                 (value ** 9) / 216)

        return 0.5 * (1 + erf)
