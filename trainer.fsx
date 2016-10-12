#load "net.fsx"

type Method = 
  | Sgd
  | Adam
  | Adagrad
  | Adadelta
  | Windowgrad
  | Netsterov


type Options = {
  learningRate : float
  l1Decay : float
  l2Decay : float
  method : Method
  momentum : float
  ro : float
  eps : float
  beta1 : float
  beta2 : float

}


open Net

let train (opt:Options) (net:Net) x y = 
  Net.forward net x true