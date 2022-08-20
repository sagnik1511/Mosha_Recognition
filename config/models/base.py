model = [
  ["input", (128, 128, 3)],
  ["conv", [16, 7, 1, "same", "relu"]],   # 122,122,16
  ["maxpool", [2]],                       # 61,61,16
  ["conv", [32, 5, 1, "same", "relu"]],   # 57,57,32
  ["maxpool", [2]],                       # 28,28,32
  ["dropout", [0.2]],
  ["conv", [128, 3, 1, "same", "relu"]],  # 26,26,128
  ["conv", [256, 3, 1, "same", "relu"]],  # 24,24,256
  ["maxpool", [2]],                       # 12,12,256
  ["dropout", [0.2]],
  ["conv", [1024, 3, 1, "same", "relu"]], # 10,10,1024
  ["dropout", [0.3]],
  ["maxpool", [2]],
  ["flatten"],                            # 102400
  ["dense", [1024, "relu"]],              # 1024
  ["dense", [1024, "relu"]],              # 1024
  ["dense", [512, "relu"]],               #512
["dense",[3, "softmax"]],                 # final number of chananels
]