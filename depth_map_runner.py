# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

import MalmoPython
import random
import time
import struct
import socket
import os
import sys
import numpy as np
from PIL import Image
from policy import Policy


# create a file handler
#handler = logging.FileHandler('depthmaprunner.log')
#handler.setLevel(logging.DEBUG)

# create a logging format
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)

# add the handlers to the logger
#logger.addHandler(handler)

#-------------------------------------------------------------------------------------------------------------------------------------

def processFrame( frame ):
    frame = np.reshape(frame, (video_height, video_width, 3))
    img = Image.fromarray(frame).convert('L')
    f = np.array(img, dtype=np.uint8)
    return f

def randomAction():
    return (np.random.randn() - 0.5) / 5

def discountRewards(r, discount):
    for i in range(len(r) - 1):
        r[len(r) - i - 2] += r[len(r) - i - 1] * discount
    return r

def step(action):
    agent_host.sendCommand( "turn " + str(action) )
    world_state = agent_host.getWorldState()
    ss = np.zeros(shape=(video_height, video_width))
    while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
        #logger.info("Waiting for frames...")
        time.sleep(0.05)
        world_state = agent_host.getWorldState()
    #logger.info("Got frame!")
    if len(world_state.video_frames) > 0:
        ss = processFrame(world_state.video_frames[0].pixels)
    r = sum(r.getValue() for r in world_state.rewards)
    done = world_state.is_mission_running
    return ss, r, done


    
#----------------------------------------------------------------------------------------------------------------------------------

video_width = 300
video_height = 150
   
missionXML = '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    
      <About>
        <Summary>Run the maze!</Summary>
      </About>
      
     <ServerSection>
        <ServerInitialConditions>
            <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
            <Weather>clear</Weather>
            <AllowSpawning>false</AllowSpawning>
        </ServerInitialConditions>
        <ServerHandlers>
            <FileWorldGenerator src="C:\\Users\\Victor\\Desktop\\MachineLearning\\RL\\Minecraft\\Malmo-0.16.0-Windows-64bit\\Python_Examples\\env" />
            <ServerQuitFromTimeUp timeLimitMs="10000"/>
            <ServerQuitWhenAnyAgentFinishes />
        </ServerHandlers>
    </ServerSection>

    <AgentSection>
        <Name>Jason Bourne</Name>
        <AgentStart>
            <Placement x="0.5" y="5" z="0.5" yaw="90" pitch="30"/>
        </AgentStart>
        <AgentHandlers>
            <VideoProducer want_depth="false">
                <Width>''' + str(video_width) + '''</Width>
                <Height>''' + str(video_height) + '''</Height>
            </VideoProducer>
            <ContinuousMovementCommands turnSpeedDegs="720" />
            <RewardForTouchingBlockType>
                <Block reward="-100.0" type="snow" behaviour="oncePerBlock"/>
                <Block reward="1000.0" type="stained_hardened_clay" behaviour="oncePerBlock"/>
                <Block reward="1000.0" type="glowstone" behaviour="oncePerBlock"/>
            </RewardForTouchingBlockType>
            <AgentQuitFromTouchingBlockType>
                <Block type="redstone_block"/>
            </AgentQuitFromTouchingBlockType>
        </AgentHandlers>
    </AgentSection>
  </Mission>'''

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

validate = True
my_mission = MalmoPython.MissionSpec( missionXML, validate )

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)

agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

e = 1
num_reps = 1000
totalSteps = 0

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 30000


PG = Policy()
rewards = []
for i_episode in range(num_reps):

    my_mission_record = MalmoPython.MissionRecordSpec()

    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                logger.error("Error starting mission: %s" % e)
                exit(1)
            else:
                time.sleep(2)

    #logger.info('Mission %s', i_episode)
    #logger.info("Waiting for the mission to start")
    world_state = agent_host.getWorldState()
    while not world_state.is_mission_running:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
    print

    agent_host.sendCommand( "move 0.5" )
    loss, R, rh = 0, 0, []
    if e > 0.01:
        e *= 0.98
    history = []
    while len(world_state.video_frames) == 0:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
    s = processFrame(world_state.video_frames[0].pixels)
    # main loop:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.is_mission_running:
            while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
                time.sleep(0.05)
                world_state = agent_host.getWorldState()
            if len(world_state.video_frames) > 0:
                s = processFrame(world_state.video_frames[0].pixels)
            #Select action
            action = randomAction()
            if(random.random() <= e):
                action = randomAction()
            else:
                action = PG.getAction(s)  
                #print action, sum(sum(s))
            ss, r, done = step(action) #sends action and waits for next frame to get reward
            R += r
            rh.append(r)
            #remeber this state and action for later training
            history.append([s, action])
            #train model
            s = ss
            totalSteps += 1
            if done:
                rewards.append(R)
    if len(rewards) > 100:
        rewards.pop(0)
    discountRewards(rh, 0.99)
    loss += PG.trainModel(history, R - sum(rewards) / len(rewards))
    filename = 'modelweights_%i.h5' % i_episode
    if i_episode % 100 == 0:
        PG.model.save_weights(filename)
    print "Iteration %i finished with reward %i, average rewards %i, epsilon %f" % (i_episode, R, sum(rewards) / len(rewards), e)

    #logger.info("Mission has stopped.")
    time.sleep(1) # let the Mod recover
