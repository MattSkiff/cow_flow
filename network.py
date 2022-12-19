#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forming a class for model definition and training method to enable ray to parallelise model training
https://docs.ray.io/en/releases-1.11.0/ray-core/using-ray-with-pytorch.html
In progress
"""

class Network(object):
    
    def __init__(self,):
        pass
        
        
    def train(self):
        pass
    
    def get_weights(self):
        pass
    
    def set_weights(self,weights):
        pass
    
    def save(self):
        pass
        
net = Network()

RemoteNetwork = ray.remote(num_gpus=1)(Network)

NetworkActor = RemoteNetwork.remote()
NetworkActor2 = RemoteNetwork.remote()

ray.get([NetworkActor.train.remote(), NetworkActor2.train.remote()])
        
        

        

