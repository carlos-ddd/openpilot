# op_params

This repo is just a backup of op_params. To install into your own fork, just grab the files in the [`openpilot`](/openpilot) folder and place them into your repository in their respective directories.

1. Make sure you add your new parameter to [`op_params.py`](/openpilot/common/op_params.py) filling out the `default`, `allowed_types`, and `live` keys. **(Update: allowed types, description, and live are no longer required but recommended if you want to give your users the ability to change parameters safely and easily. Live is assumed to be False by default if not specified. `default` is still required!)**
   * You can also change the read, or update frequency in that file for live tuning.
2. In the file you want to receive updates, use this code in place of the variable you want to use.
   * So for camera offset, we can do this in `lane_planner.py`:
   ```python
   from common.op_params import opParams
   op_params = opParams()
   CAMERA_OFFSET = op_params.get('camera_offset', default=0.06)  # this is not live tunable
   ```
   * The default will be used if the user doesn't have the parameter.
3. **Important**: for variables you want to be live tunable, you need to use the `op_params.get()` function to set the variable on each update. So for example, with classes, you need to initialize the variable in the `__init__` function, and then in the update function, set it again. Here's a fake example for longcontrol.py:
```python
from common.op_params import opParams


class LongControl():
  def __init__():
    self.op_params = opParams()  # always init opParams first
    self.your_live_tunable_variable = self.op_params.get('whatever_param', default=0.5)  # initializes the variable
  
  def update(stuff...):
    self.your_live_tunable_variable = self.op_params.get('whatever_param', default=0.5)  # and this updates it as you tune. will not update live if you don't occasionally call .get() on your parameter.
    
    # then we use the variable down here... 
```

4. Now to change live parameters over ssh, you can connect to your EON with your WiFi hotspot, then change directory to `/data/openpilot` and run `python op_edit.py` (which now fully supports live tuning). It's important to make sure you set `'live'` to `True` for any parameters you want to be live.
   * Here's a gif of the tuner:

<img src="gifs/op_tune.gif?raw=true" width="600">
