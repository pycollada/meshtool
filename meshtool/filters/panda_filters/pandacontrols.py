from direct.showbase.DirectObject import DirectObject
from direct.task import Task
import sys

class KeyboardMovement(DirectObject):
    def __init__(self):

        #Controls how fast movement is        
        self.SPEED = 10
        
        #initial values for movement of the camera
        self.cam_pos_x = 0
        self.cam_pos_y = 0
        self.cam_pos_z = 0
        self.cam_h = 0
        self.cam_p = 0
        self.cam_r = 0
        
        #Each input modifies x,y,z, h,p,r by an amount
        key_modifiers = [('w', [True, 0, 0.5, 0, 0, 0, 0]),
                         ('s', [True, 0, -0.5, 0, 0, 0, 0]),
                         ('a', [True, -0.5, 0, 0, 0, 0, 0]),
                         ('d', [True, 0.5, 0, 0, 0, 0, 0]),
                         ('r', [True, 0, 0, 0.5, 0, 0, 0]),
                         ('f', [True, 0, 0, -0.5, 0, 0, 0]),
                         ('arrow_up', [True, 0, 0.5, 0, 0, 0, 0]),
                         ('arrow_down', [True, 0, -0.5, 0, 0, 0, 0]),
                         ('arrow_left', [True, 0, 0, 0, 5, 0, 0]),
                         ('arrow_right', [True, 0, 0, 0, -5, 0, 0])]
        
        for key, extra_args in key_modifiers:
            unpress = list(extra_args)
            unpress[0] = False
            self.accept(key, self.keypress, extraArgs=extra_args)
            self.accept(key + '-up', self.keypress, extraArgs=unpress)
        
        self.accept('escape', sys.exit)
        
        movingTask = taskMgr.add(self.moving, "movingTask")
        movingTask.last = 0 
        
    def moving(self, task):
        # Standard technique for finding the amount of time since the last frame
        dt = task.time - task.last
        task.last = task.time
        
        scale = dt * self.SPEED
        base.cam.setPosHpr(base.cam,
                           scale * self.cam_pos_x,
                           scale * self.cam_pos_y,
                           scale * self.cam_pos_z,
                           scale * self.cam_h,
                           scale * self.cam_p,
                           scale * self.cam_r)
        return Task.cont
        
    def keypress(self, down, x, y, z, h, p, r):
        scale = 1 if down else -1
        self.cam_pos_x += scale * x
        self.cam_pos_y += scale * y
        self.cam_pos_z += scale * z
        self.cam_h += scale * h
        self.cam_p += scale * p
        self.cam_r += scale * r
