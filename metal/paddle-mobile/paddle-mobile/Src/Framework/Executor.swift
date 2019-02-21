/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

import Foundation


let testTo = 5

var isTest = false

@objc public class GPUResultHolder: NSObject{
  @objc public let dim: [Int]
  @objc public let capacity: Int
  @objc public var resultPointer: UnsafeMutablePointer<Float32>?
  @objc public var intermediateResults: [String : [MTLBuffer]]?
  public init(inDim: [Int], inPointer: UnsafeMutablePointer<Float32>?, inCapacity: Int, inIntermediateResults: [String : [MTLBuffer]]? = nil) {
    dim = inDim
    capacity = inCapacity
    
    if let inInPointer = inPointer {
      resultPointer = UnsafeMutablePointer<Float32>.allocate(capacity: inCapacity)
      resultPointer?.initialize(from: inInPointer, count: inCapacity)
    }
    
    intermediateResults = inIntermediateResults
  }
  
  public override var description: String {
    fatalError()
  }
  
}

public class Executor<P: PrecisionType> {
  var ops: [Runable & InferShaperable] = []
  var preInputDim: Dim = Dim.init(inDim: [])
  let program: Program
  let device: MTLDevice
  let inflightSemaphore: DispatchSemaphore
  let queue: MTLCommandQueue
  init(inDevice:MTLDevice, inQueue: MTLCommandQueue, inProgram: Program, initContext: InitContext) throws {
    self.inflightSemaphore = DispatchSemaphore(value: 1)
    program = inProgram
    device = inDevice
    queue = inQueue
    
    for block in inProgram.programDesc.blocks {
      //block.ops.count
      for i in 0..<block.ops.count {
        let opDesc = block.ops[i]
        do {
          let op = try OpCreator<P>.shared.creat(device: inDevice, opDesc: opDesc, scope: inProgram.scope, initContext: initContext)
          ops.append(op)
        } catch let error {
          throw error
        }
      }
    }
  }
  
  public func predict(input: MTLTexture, dim: Dim, completionHandle: @escaping ([GPUResultHolder]) -> Void, preProcessKernle: CusomKernel? = nil, except: Int = 0) throws {
    inflightSemaphore.wait()

    guard let buffer = queue.makeCommandBuffer() else {
      throw PaddleMobileError.predictError(message: "CommandBuffer is nil")
    }
    
    let resInput: MTLTexture
    if let inPre = preProcessKernle {
      do {
        try inPre.compute(inputTexuture: input, commandBuffer: buffer)
        resInput = inPre.outputTexture
      } catch let error {
        throw error
      }
    } else {
      resInput = input
    }
    
    let inputTexture = InputTexture.init(inMTLTexture: resInput, inExpectDim: dim)
    program.scope.setInput(input: inputTexture)
    //(ops.count - except)
    for i in 0..<(ops.count - except) {
      let op = ops[i]
      do {
        try op.run(device: device, buffer: buffer)
      } catch let error {
        throw error
      }
    }
    
    var outputTextures: [String : [MTLBuffer]]?
    if except > 0 {
      ops[ops.count - except].computeMiddleResult(device: device, buffer: buffer)
      outputTextures = ops[ops.count - except].inputVariant()
    }
    
    buffer.addCompletedHandler { [weak self] (commandbuffer) in
      guard let SSelf = self else {
        fatalError()
      }
            
      //将输入写进文件
      /*
       let inputArr = resInput.toTensor(dim: (n: dim[0], c: dim[3], h: dim[1], w: dim[2]))
       print(dim)
       writeToLibrary(fileName: "test_image_super", array: inputArr)
       print(" write done ")
       return
       */
      
      /*    输出 op 计算结果
       for op in SSelf.ops {
       op.delogOutput()
       }
       */
      
      var resultHolder: GPUResultHolder
      if except > 0 {
        resultHolder = GPUResultHolder.init(inDim: [], inPointer: nil, inCapacity: 0,  inIntermediateResults: outputTextures)
      } else {
        let outputVar: Variant = SSelf.program.scope.output()!
        let output: FetchHolder = outputVar as! FetchHolder
        resultHolder = GPUResultHolder.init(inDim: output.dim.dims, inPointer: output.result, inCapacity: output.capacity)
      }
      
      completionHandle([resultHolder])
      SSelf.inflightSemaphore.signal()
    }
    
    buffer.commit()
  }
  
  public func clear() {
    program.scope.clear()
  }
  
}
