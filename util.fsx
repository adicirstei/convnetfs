type MaxMin = {
  maxi : int
  maxv : float
  mini : int
  minv : float
  dv : float
}

let rndg = System.Random()

let swap (a: _[]) x y =
    let tmp = a.[x]
    a.[x] <- a.[y]
    a.[y] <- tmp

// shuffle an array (in-place)
let shuffle a =
    Array.iteri (fun i _ -> swap a i (rndg.Next(i, Array.length a))) a

let rec gaussRandom () =
  let u = 2.0 * rndg.NextDouble() - 1.0
  let v = 2.0 * rndg.NextDouble() - 1.0
  let r = v*v + u*u
  
  if r = 0.0 || r > 1.0 then gaussRandom()
  else 
    let c = sqrt ( -2.0 * (log r) / r)
    u * c
  
let randf a b = rndg.NextDouble() * (b - a) + a
let randi a b = randf a b |> int |> float
let randn mu std = mu + gaussRandom() * std 

let zeros = Array.zeroCreate<float>

let arrContains = Array.contains

let arrUnique = Array.distinct

let maxmin = 
  Array.mapi (fun i v -> i,v ) 
  >>
  Array.fold (fun st (i,v) -> 
            match st with 
            | None -> Some { maxi = i; mini = i; maxv = v; minv = v; dv = 0.0}
            | Some mm -> Some { mm with 
                                  maxi = if v > mm.maxv then i else mm.maxi
                                  mini = if v < mm.minv then i else mm.mini
                                  maxv = if v > mm.maxv then v else mm.maxv 
                                  minv = if v < mm.minv then v else mm.minv 
                                  dv = (if v > mm.maxv then v else mm.maxv) - (if v < mm.minv then v else mm.minv)}
          ) None 

let randperm n = 
  let a = [|0..(n-1)|]
  shuffle a
  a

/// sample from list lst according to probabilities in list probs
/// the two lists are of same size, and probs adds up to 1
let weightedSample (lst:'a []) probs = 
  let p = randf 0.0 1.0
  let idx = Array.scan (+) 0.0 probs
            |> Array.takeWhile ((>) p)
            |> Array.length
  lst.[idx-1]

