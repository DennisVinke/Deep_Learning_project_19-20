import pygame
import torch
import sys
import functools
from torch import nn
from model import ResnetGenerator
from torchvision import transforms
from skimage import color  # used for lab2rgb
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import init_net
from models.cycle_gan_model import CycleGANModel
from options.test_options import TestOptions
from models import create_model
import os


class DrawingCanvas:


    def __init__(self):
        opt = TestOptions().parse()  # get test options
        # init pygame
        pygame.init()
        self.size = (256, 256)
        self.screen = pygame.display.set_mode(self.size)
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0], 64)
        self.time = pygame.time.get_ticks()
        #self.surface_test = pygame.surfarray.make_surface()
        self.screen.fill(pygame.Color(255, 255, 255))
        pygame.display.flip()

        self.model = CycleGANModel(opt)
        self.model.setup(opt)
        #norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        #net = ResnetGenerator(256, 256, 64, norm_layer=norm_layer, use_dropout=False, n_blocks=9)
        #self.net = init_net(net, 'normal', 0.02, [])
        impath = os.getcwd()+"/datasets/bird/testA/514.png"

        image = pygame.image.load(impath)
        #self.screen.blit(image, (0, 0))

    """
    Method 'game_loop' will be executed every frame to drive
    the display and handling of events in the background. 
    In Processing this is done behind the screen. Don't 
    change this, unless you know what you are doing.
    """
    def game_loop(self):
        current_time = pygame.time.get_ticks()
        delta_time = current_time - self.time
        self.time = current_time
        self.handle_events()
        self.update_game(delta_time)
        self.draw_components()

    """
    Method 'update_game' is there to update the state of variables 
    and objects from frame to frame.
    """
    def update_game(self, dt):
        pass

    """
    Method 'draw_components' is similar is meant to contain 
    everything that draws one frame. It is similar to method
    void draw() in Processing. Put all draw calls here. Leave all
    updates in method 'update'
    """
    def draw_components(self):
        #self.screen.fill([255, 255, 255])
        #pygame.display.flip()
        pass

    def reset(self):
        pass

    """
    Method 'handle_event' loop over all the event types and 
    handles them accordingly. 
    In Processing this is done behind the screen. Don't 
    change this, unless you know what you are doing.
    """
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                self.handle_key_down(event)
            if event.type == pygame.KEYUP:
                self.handle_key_up(event)
            if event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_pressed(event)
            if event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_released(event)

    """
    This method will store a currently pressed buttons 
    in list 'keyboard_handler.pressed'.
    """
    def handle_key_down(self, event):
        pass

    """
    This method will remove a released button 
    from list 'keyboard_handler.pressed'.
    """
    def handle_key_up(self, event):
        pass

    """
    Similar to void mouseMoved() in Processing
    """
    def handle_mouse_motion(self, event):
        #print("test: ",pygame.mouse.get_pressed()[0])
        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            pygame.display.update(pygame.draw.ellipse(self.screen, (0, 0, 0), [pos, [5, 5]]))
            #print(pos)
            self.screen.blit(self.screen, (0, 0))

    """
    Similar to void mousePressed() in Processing
    """
    def handle_mouse_pressed(self, event):
        pos = pygame.mouse.get_pos()
        pygame.display.update(pygame.draw.rect(self.screen, (0, 0, 0), [pos, [5,5]]))
        #(pos)
        self.screen.blit(self.screen, (0, 0))
    """
    Similar to void mouseReleased() in Processing
    """
    def handle_mouse_released(self, event):
        #pygame.display.flip()
        test = pygame.surfarray.array3d(self.screen)

        print(test.shape)
        #test = test.T
        test = test.transpose(1,0,2)
        print(test.shape)
        #string_image = pygame.image.tostring(self.screen, 'RGBA')
        #temp_surf = pygame.image.fromstring(string_image, (512, 512), 'RGB')
        #tmp_arr = pygame.surfarray.array2d(temp_surf)

        compose = transforms.Compose([transforms.ToPILImage(),
                                      #transforms.Resize(256, interpolation=Image.CUBIC),
                                      transforms.ToTensor()])

        test_tensor = compose(test).unsqueeze(0)
        #plt.figure()
        #plt.imshow(test)
        #plt.show()


        print(test_tensor.size())
        #test = compose(test)
        #self.net.set_input(test)
        self.model.set_input(test_tensor)
        result = self.model.forward()
        result = self.model.get_generated()
        print("Result", result)

        resultT = result.squeeze(0)
        resultT[resultT<0]=0
        im = transforms.ToPILImage()(resultT).convert("RGB")


        test = result.squeeze(0)
        print(test.size())
        result = result.detach().numpy()
        result = np.squeeze(result, axis=0)
        result = result.transpose(1,2,0)
        print(result.shape)
        results = result[:]*255
        #results[result < 0]
        print(im)
        print(im.size)
        plt.imshow(im)
        plt.show()



    def lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)
        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb


if __name__ == "__main__":
    canvas = DrawingCanvas()
    while True:
        canvas.game_loop()