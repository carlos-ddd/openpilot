from cereal import log
from common.numpy_fast import clip, interp
from selfdrive.controls.lib.pid import PIController
from common.op_params import opParams

LongCtrlState = log.ControlsState.LongControlState

STOPPING_EGO_SPEED = 0.5
MIN_CAN_SPEED = 0.3  # TODO: parametrize this in car interface
STOPPING_TARGET_SPEED = MIN_CAN_SPEED + 0.01
STARTING_TARGET_SPEED = 0.5
BRAKE_THRESHOLD_TO_PID = 0.2

STOPPING_BRAKE_RATE = 0.2  # brake_travel/s while trying to stop
STARTING_BRAKE_RATE = 0.8  # brake_travel/s while releasing on restart
BRAKE_STOPPING_TARGET = 0.5  # apply at least this amount of brake to maintain the vehicle stationary

RATE = 100.0


def long_control_state_trans(active, long_control_state, v_ego, v_target, v_pid,
                             output_gb, brake_pressed, cruise_standstill):
  """Update longitudinal control state machine"""
  stopping_condition = (v_ego < 2.0 and cruise_standstill) or \
                       (v_ego < STOPPING_EGO_SPEED and
                        ((v_pid < STOPPING_TARGET_SPEED and v_target < STOPPING_TARGET_SPEED) or
                        brake_pressed))

  starting_condition = v_target > STARTING_TARGET_SPEED and not cruise_standstill

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if active:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.starting

    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif output_gb >= -BRAKE_THRESHOLD_TO_PID:
        long_control_state = LongCtrlState.pid

  return long_control_state


class LongControl():
  def __init__(self, CP, compute_gb):
    self.op_params = opParams()     # for live parameter tuning of longitudinal (carlos-ddd)
    self.long_control_state = LongCtrlState.off  # initialized to off
    self.pid = PIController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                            (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                            rate=RATE,
                            sat_limit=0.8,
                            convert=compute_gb)
    self.update_liveParams(CP)    # overwrite those of the line above (carlos-ddd)
    self.v_pid = 0.0
    self.last_output_gb = 0.0

    self.prntCount = 0
    self.prntLoop = 0
    self.prntTotal = 0
    self.prntGas = []
    self.prntBrake = []
    self.prntVist = []
    self.prntVsoll = []
    self.prntGB = []


  def reset(self, v_pid):
    """Reset PID controller and change setpoint"""
    self.pid.reset()
    self.v_pid = v_pid

  def update_liveParams(self, CP):  # carlos-ddd
  
    if self.op_params.get('long_tune_single_params') == True:  # spot tuning
      self.pid._k_p = ([self.op_params.get('kpBP')], [self.op_params.get('kpV')])
      self.pid._k_i = ([self.op_params.get('kiBP')], [self.op_params.get('kiV')])
    else:  # use interpolated (final list)
      #       kph:      10   30   50    80    120
      self.pid._k_p = ([2.8, 8.3, 13.8, 22.2, 33.3], [self.op_params.get('kpV_10'), self.op_params.get('kpV_30'), self.op_params.get('kpV_50'), self.op_params.get('kpV_80'), self.op_params.get('kpV_120')])
      self.pid._k_i = ([2.8, 8.3, 13.8, 22.2, 33.3], [self.op_params.get('kiV_10'), self.op_params.get('kiV_30'), self.op_params.get('kiV_50'), self.op_params.get('kiV_80'), self.op_params.get('kiV_120')])
    # self.pid.reset() is done within the call of LongControl.update()->"LongCtrlState.off or CS.gasPressed" call-path
    
    # FOLLOWING values have no breakpoints yet!
    CP.gasMaxBP = [0., self.op_params.get('gasMaxBP')]
    CP.gasMaxV = [.3, self.op_params.get('gasMaxV')]  # limit maximum gas in standstill (safety)
    
    CP.brakeMaxBP = [self.op_params.get('brakeMaxBP')]
    CP.brakeMaxV = [self.op_params.get('brakeMaxV')]
    
    CP.longitudinalTuning.deadzoneBP = [self.op_params.get('deadzoneBP')]
    CP.longitudinalTuning.deadzoneV = [self.op_params.get('deadzoneV')]
    

  def update(self, active, CS, v_target, v_target_future, a_target, CP):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    # Actuation limits
    gas_max = interp(CS.vEgo, CP.gasMaxBP, CP.gasMaxV)
    brake_max = interp(CS.vEgo, CP.brakeMaxBP, CP.brakeMaxV)

    # Update state machine
    output_gb = self.last_output_gb
    self.long_control_state = long_control_state_trans(active, self.long_control_state, CS.vEgo,
                                                       v_target_future, self.v_pid, output_gb,
                                                       CS.brakePressed, CS.cruiseState.standstill)

    v_ego_pid = max(CS.vEgo, MIN_CAN_SPEED)  # Without this we get jumps, CAN bus reports 0 when speed < 0.3

    output_gb_save = 0. # carlos-ddd

    if self.long_control_state == LongCtrlState.off or CS.gasPressed:
      self.update_liveParams(CP)    # carlos-ddd
      self.reset(v_ego_pid)
      output_gb = 0.

    # tracking objects and driving
    elif self.long_control_state == LongCtrlState.pid:
      self.v_pid = v_target
      self.pid.pos_limit = gas_max
      self.pid.neg_limit = - brake_max

      # Toyota starts braking more when it thinks you want to stop
      # Freeze the integrator so we don't accelerate to compensate, and don't allow positive acceleration
      prevent_overshoot = not CP.stoppingControl and CS.vEgo < 1.5 and v_target_future < 0.7
      deadzone = interp(v_ego_pid, CP.longitudinalTuning.deadzoneBP, CP.longitudinalTuning.deadzoneV)

      output_gb = self.pid.update(self.v_pid, v_ego_pid, speed=v_ego_pid, deadzone=deadzone, feedforward=a_target, freeze_integrator=(prevent_overshoot or CS.clutchPressed))
      output_gb_save = output_gb # carlos-ddd save for later plotting before clipping, limiting, etc to evaluate pid-performance

      if prevent_overshoot:
        output_gb = min(output_gb, 0.0)

    # Intention is to stop, switch to a different brake control until we stop
    elif self.long_control_state == LongCtrlState.stopping:
      # Keep applying brakes until the car is stopped
      if not CS.standstill or output_gb > -BRAKE_STOPPING_TARGET:
        output_gb -= STOPPING_BRAKE_RATE / RATE
      output_gb = clip(output_gb, -brake_max, gas_max)

      self.reset(CS.vEgo)

    # Intention is to move again, release brake fast before handing control to PID
    elif self.long_control_state == LongCtrlState.starting:
      if output_gb < -0.2:
        output_gb += STARTING_BRAKE_RATE / RATE
      self.reset(CS.vEgo)

    self.last_output_gb = output_gb
    final_gas = clip(output_gb, 0., gas_max)
    final_brake = -clip(output_gb, -brake_max, 0.)
   
    # carlos-ddd plotting efforts
    self.prntLoop += 1
    if self.prntLoop >= 20: # record every 20th value => 5 values per second (100Hz calls here)
      self.prntLoop = 0
      self.prntGas.append(final_gas)
      self.prntBrake.append(final_brake)
      self.prntVist.append(v_ego_pid)
      self.prntVsoll.append(self.v_pid)
      self.prntGB.append(output_gb_save)
      self.prntCount += 1
      if self.prntCount >= 10: # print them every 2 seconds (10 values at once)
        self.prntCount = 0
        print("k2-plot:%i:"%(self.prntTotal), end='')
        
        for itm in self.prntGas:
          print("%.5f,"%(itm), end='')
        print(":", end='')
        
        for itm in self.prntBrake:
          print("%.5f,"%(itm), end='')
        print(":", end='')
        
        for itm in self.prntVist:
          print("%.2f,"%(itm), end='')
        print(":", end='')
        
        for itm in self.prntVsoll:
          print("%.1f,"%(itm), end='')
        print(":", end='')
        print(";")
        
        self.prntGas.clear()
        self.prntBrake.clear()
        self.prntVist.clear()
        self.prntVsoll.clear()
        self.prntGB.clear()
        self.prntTotal += 1

    return final_gas, final_brake
