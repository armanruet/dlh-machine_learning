#!/usr/bin/env python3
"""creating a Class"""




class Binomial:
    """defining the class"""
    @staticmethod
    def fact(n):
        """def fact"""
        if n == 0:
            return 1
        return n * Binomial.fact(n - 1)

    def __init__(self, data=None, n=1, p=0.5):
        """initiating the constructor"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = n
            self.p = p

    def pmf(self, k):
        """defining pmf"""
        k = int(k)

        if k < 0 or k > self.n:
            return 0
        coefficient = Binomial.fact(
            self.n) / (Binomial.fact(k) * Binomial.fact(self.n - k))
        return coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))
