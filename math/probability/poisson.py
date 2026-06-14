#!/usr/bin/env python3
"""Class to represents a poisson distribution"""

class Poisson:
    """defining the class"""

    def __init__(self, data=None, lambtha=1.):
        """Initiating the Class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculating PMF"""
        k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(2, k+1):
            fact *= i
        e = 2.7182818285
        return (e ** -self.lambtha * self.lambtha ** k) / fact

