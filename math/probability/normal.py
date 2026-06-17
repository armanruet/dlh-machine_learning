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
