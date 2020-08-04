import os
import tempfile
import numpy as np
from datetime import datetime
import tensorflow as tf
from moviepy.editor import ImageSequenceClip
import gym

class GymBoard(object):
    """
    Wrapper for visualizing Gym envs in TensorBoard.
    
    Args:
        logdir (str, optional): Filesystem directory where TensorBoard logs will be written. Defaults to `logs/gymboard/`.
        port (int, optional): TensorBoard will bind to this port. Defaults to 8081.

    Attributes:
        logdir (str): Filesystem directory where TensorBoard logs will be written.
        logrun (str): Folder in `logrun` that contains current run's summaries and events.
        port (int): TensorBoard will bind to this port.
        writer (SummaryWriter): SummaryWriter that writes events and summarries in `logrun`.
        step (int): Training step/epoch number.
    """
    
    def __init__(self, logdir="logs/", port=6006):
        self.logdir = logdir
        self.logrun = self.logdir + datetime.now().strftime("%b_%d_%X") 
        self.port = port
        self.writer = tf.summary.create_file_writer(self.logrun)
        self.step = 0
    
    def write_scalar(self, label, score, step=None):
        """
        Write scalar values (e.g. reward score) to TensorBoard.
    
        Args:
            label (str): Label/Name for this summary/metric (e.g. "reward_score").
            score (int/float): Scalar value of the summary.
            step (int, optional): Current step/epoch number in the training loop. Defaults to `self.step`.
        """
        
        if step is None:
            step = self.step
            self.step += 1
        else: self.step = step
            
        with self.writer.as_default():
            tf.summary.scalar(label, score, step=step)
            self.writer.flush()
    
    def write_env(self, env, policy=None, step=None, fps=None, speed=1.0):
        """
        Writes GIF (output from `env` for single episode by following `policy` greedily) to TensorBoard. 
        
        Important: given `env` state/observation, `policy` must return tensor in shape [n, a, 1] where n=batch_size, a=env.action_space.n.

        Args:
            env (Gym Env): OpenAI Gym environment.
            policy (function, optional): Policy to be followed. Defaults to random policy.
            step (int, optional): Current step/epoch number in the training loop. Defaults to `self.step`.
            fps (int, optional): Frames per second of the resulting GIF. Defaults to `env.metadata['video.frames_per_second']`.
            speed (float, optional): Speed of the rendered GIF (e.g. speed=2.0 will increase the speed twice). Defaults to 1.0.
        """

        state = env.reset()
        steps, rewards = [], []
        while True:
            steps.append(env.render('rgb_array'))
            try:
                action = policy(np.expand_dims(state,axis=0))[0]
                action = np.argmax(action)
            except: 
                action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done: 
                break
        
        if fps is None:
            try: fps = env.metadata['video.frames_per_second']
            except: fps = 24
        fps = int(fps*speed)
        
        # from https://github.com/tensorflow/tensorboard/issues/39#issuecomment-568917607
        thwc = env.render('rgb_array').shape
        im_summ = tf.compat.v1.Summary.Image()
        im_summ.height, im_summ.width = thwc[0], thwc[1]
        im_summ.colorspace = 3 # fix to 3 for RGB
        with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
        clip = ImageSequenceClip(steps, fps=fps)
        clip.write_gif(fname, verbose=False, logger=None)
        with open(fname, 'rb') as f: enc_gif = f.read()
        os.remove(fname)
        
        im_summ.encoded_image_string = enc_gif
        # create a serialized summary obj:
        gif = tf.compat.v1.Summary()
        env_name = env.unwrapped.spec.id
        gif.value.add(image=im_summ, tag=f'{env_name}/ Rewards: {sum(rewards)}')
        
        if step is None: step = self.step
        with self.writer.as_default():
            tf.summary.experimental.write_raw_pb(gif.SerializeToString(), step=step)    
        env.close()

    def display(self):
        """
        Show TensorBoard within Notebook environment (if possible). Otherwise, print instructions on how to run TensorBoard.
        """
        try:
            mgc = get_ipython().magic
            mgc('%load_ext tensorboard')
            mgc(f'%tensorboard --logdir {self.logdir} --port={self.port} --bind_all')
        except:
            print(f'Run command below:\ntensorboard --logdir {self.logdir} --port={self.port} --bind_all')
            
    def clean(self): 
        """
        Clean `self.logdir` by removing all summaries and events from filesystem.
        """
        os.system(f'rm -rf {self.logdir}')
