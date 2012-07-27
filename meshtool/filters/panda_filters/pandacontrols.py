from direct.showbase.DirectObject import DirectObject
from direct.task import Task
from panda3d.core import Vec3, Quat
import sys
from math import pi

class KeyboardMovement(DirectObject):
    def __init__(self, scale=1.0):
        self.scale = scale
        self.adjustSpeed()
        
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
        self.accept('[', self.speedDown)
        self.accept(']', self.speedUp)
        
        movingTask = taskMgr.add(self.moving, "movingTask")
        movingTask.last = 0 
        
    def moving(self, task):
        # Standard technique for finding the amount of time since the last frame
        dt = task.time - task.last
        task.last = task.time
        
        posscale = dt * self.POS_SPEED
        camscale = dt * self.CAM_SPEED
        base.cam.setPosHpr(base.cam,
                           posscale * self.cam_pos_x,
                           posscale * self.cam_pos_y,
                           posscale * self.cam_pos_z,
                           camscale * self.cam_h,
                           camscale * self.cam_p,
                           camscale * self.cam_r)
        return Task.cont
    
    def adjustSpeed(self):
        #Controls how fast movement is        
        self.POS_SPEED = 1000 * self.scale
        #Controls how fast camera movement is
        self.CAM_SPEED = 4 * self.scale

    def speedDown(self):
        self.scale *= 0.5
        self.adjustSpeed()

    def speedUp(self):
        self.scale *= 2.0
        self.adjustSpeed()

    def keypress(self, down, x, y, z, h, p, r):
        scale = 1 if down else -1
        self.cam_pos_x += scale * x
        self.cam_pos_y += scale * y
        self.cam_pos_z += scale * z
        self.cam_h += scale * h
        self.cam_p += scale * p
        self.cam_r += scale * r

class ButtonUtils(DirectObject):
    def __init__(self, node):
        self.node = node
        self.wireframe = False
        self.accept('i', self.toggleWireframe)

    def toggleWireframe(self):
        self.wireframe = not self.wireframe
        
        if self.wireframe:
            self.node.setRenderModeWireframe()
        else:
            self.node.setRenderModeFilled()

class MouseCamera(DirectObject):
    def __init__(self):
        
        #The scale of the rotation relative to the mouse coords
        self.SCALE = 25

        self.accept("mouse3", self.down)
        self.accept("mouse3-up", self.up)
        
    def down(self):
        if not base.mouseWatcherNode.hasMouse():
            return

        self.initialMouseCoord = (base.mouseWatcherNode.getMouseX(), base.mouseWatcherNode.getMouseY())
        self.initialH = base.cam.getH()
        self.initialP = base.cam.getP()
        
        taskMgr.add(self.drag, "rightdrag")
    
    def drag(self, task):
        if not base.mouseWatcherNode.hasMouse():
            return Task.cont
        
        curMouseCoord = (base.mouseWatcherNode.getMouseX(), base.mouseWatcherNode.getMouseY())
        delta = (curMouseCoord[0] - self.initialMouseCoord[0], curMouseCoord[1] - self.initialMouseCoord[1])
        
        base.cam.setH(self.initialH + -1 * self.SCALE * delta[0])
        base.cam.setP(self.initialP + self.SCALE * delta[1])
        
        return Task.cont
    
    def up(self):
        taskMgr.remove("rightdrag")

class MouseDrag(DirectObject):
    def __init__(self, node):
        
        #The scale of the rotation relative to the mouse coords
        self.SCALE = 0.4*pi * (180.0 / pi)
        
        self.node = node
        self.accept("mouse1", self.down)
        self.accept("mouse1-up", self.up)
        
    def down(self):
        if not base.mouseWatcherNode.hasMouse():
            return

        self.initialMouseCoord = (base.mouseWatcherNode.getMouseX(), base.mouseWatcherNode.getMouseY())
        self.initialQuat = self.node.getQuat(base.cam)
        
        taskMgr.add(self.drag, "drag")
    
    def drag(self, task):
        if not base.mouseWatcherNode.hasMouse():
            return Task.cont
        
        curMouseCoord = (base.mouseWatcherNode.getMouseX(), base.mouseWatcherNode.getMouseY())
        delta = (curMouseCoord[0] - self.initialMouseCoord[0], curMouseCoord[1] - self.initialMouseCoord[1])
        
        xQuat = Quat()
        xQuat.setFromAxisAngle(delta[0] * self.SCALE, Vec3(0,0,1))
        
        yQuat = Quat()
        yQuat.setFromAxisAngle(delta[1] * self.SCALE * -1, Vec3(1,0,0))
        
        self.node.setQuat(base.cam, self.initialQuat * xQuat * yQuat)
        
        return Task.cont
    
    def up(self):
        taskMgr.remove("drag")

class MouseScaleZoom(DirectObject):
    def __init__(self, node):
        self.node = node
        self.scale_increment = self.node.getScale() * 0.10
        self.accept("wheel_up", self.up)
        self.accept("wheel_down", self.down)
    def up(self):
        self.node.setScale(self.node.getScale() - self.scale_increment)
        if self.node.getScale() < self.scale_increment:
            self.node.setScale(self.scale_increment)
    def down(self):
        self.node.setScale(self.node.getScale() + self.scale_increment)
