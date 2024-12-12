# -*- coding: utf-8 -*-
"""
Created on Tue June 24 10:16:00 2024
 
@author: tom.wh.cheng
"""


class ParameterModify:
    def __init__(self, step_size: float = 0.01):
        self.step_size = step_size
        self.mse_list = []

    def modify(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        mse: float,
        integral_error: float,
        ov_upper: float,
        ov_lower: float,
    ):
        # provide current Kp Ki Kd
        # provide tail 30 mse
        # provide tail 30 integral error sum

        if mse / abs(ov_upper - ov_lower) * 30 < 0.3:
            pass
        else:
            if integral_error**2 > mse:
                if integral_error < 0:
                    Kp *= 1 + self.step_size

                else:
                    Kp *= 1 - self.step_size
                    pass
            else:
                Ki *= 1 + self.step_size / 30

        return Kp, Ki, Kd
