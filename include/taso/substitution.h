/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _SUBSTITUTION_H_
#define _SUBSTITUTION_H_
#include "taso/ops.h"
#include "rules.pb.h"
#include <queue>
namespace taso {

enum Compare {
  COMPARE_EQ,
  COMPARE_NE,
  COMPARE_LT,
  COMPARE_LE,
  COMPARE_GT,
  COMPARE_GE,
};

struct PMConstraint {
  PMConstraint(Compare comp, PMParameter para, int value);
  Compare comp;
  PMParameter para;
  int value;
};

struct TNConstraint {
  TNConstraint(Compare comp, TNParameter para, DIMParameter dim, int value);
  TNConstraint(Compare comp, TNParameter para1, DIMParameter dim1,
               TNParameter para2, DIMParameter dim2);
  bool singlePara;
  Compare comp;
  TNParameter para1, para2;
  DIMParameter dim1, dim2;
  int value;
};

class OpX;
class GraphXfer;
struct TensorX {
  TensorX(void): op(NULL), idx(0) {}
  TensorX(std::shared_ptr<OpX> _op, int _idx): op(_op), idx(_idx) {}
  Tensor to_tensor(const std::shared_ptr<GraphXfer> xfer) const;
  std::shared_ptr<OpX> op;
  int idx;
};

struct TensorXCompare {
  bool operator()(const TensorX& a, const TensorX& b) const {
    if (a.op != b.op) return a.op < b.op;
    return a.idx < b.idx;
  };
};

class OpX : std::enable_shared_from_this<OpX> {
public:
  OpX(const OpX& _op);
  OpX(OpType _type, TensorX input0, int numOutputs = 1);
  OpX(OpType _type, TensorX input0, TensorX input1);
  OpX(OpType _type, TensorX input0, TensorX input1, TensorX input2);
  OpX(OpType _type, TensorX input0, TensorX input1, TensorX input2, TensorX input3);
  OpX(OpType _type, TensorX input0, TensorX input1, TensorX input2, TensorX input3, TensorX input4);
  OpX(OpType _type, int n, TensorX* ins);
  bool add_pm_constraint(Compare comp, PMParameter para, int value);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, int);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, TNParameter, DIMParameter);
  bool get_pm_constraint(PMParameter para, int& value) const;
public:
  OpType type;
  Op mapOp;
  std::vector<TensorX> inputs, outputs;
  std::vector<PMConstraint> pmConstraints;
  std::vector<TNConstraint> tnConstraints;
};

class DstOp;
class SrcOp {
public:
  SrcOp(OpType _type);
  bool add_constraint(Compare comp, PMParameter para, int value);
  bool match(Op op);
public:
  std::vector<PMConstraint> constraints;
  OpType type;
  Op mapOp;
  std::shared_ptr<DstOp> mapInput, mapOutput;
};

class DstOp {
public:
  DstOp(OpType _type);
  DstOp(OpType _type, const std::shared_ptr<SrcOp> op);
  DstOp(OpType _type, const std::shared_ptr<SrcOp> op1, const std::shared_ptr<SrcOp> op2);
  virtual Op create_operator(std::shared_ptr<Model> model) = 0;
public:
  OpType type;
  Op mapOp;
  std::shared_ptr<SrcOp> mapInput, mapOutput;
  std::shared_ptr<SrcOp> srcOps[MAX_NUM_INPUTS];
};

template <typename OpType>
struct SubEdge {
  SubEdge(std::shared_ptr<OpType> _srcOp, std::shared_ptr<OpType> _dstOp, int _srcIdx, int _dstIdx)
  : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx) {}
  int srcIdx, dstIdx;
  std::shared_ptr<OpType> srcOp, dstOp;
};

template<typename OpType>
struct SubEdgeCompare {
  bool operator()(const SubEdge<OpType>& a, const SubEdge<OpType>& b) const {
    if (a.srcOp != b.srcOp) return a.srcOp < b.srcOp;
    if (a.dstOp != b.dstOp) return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
    return false;
  };
};

class GraphCompare {
public:
  bool operator() (std::shared_ptr<Graph> lhs, std::shared_ptr<Graph> rhs) {
    return lhs->total_cost() > rhs->total_cost();
  }
};

class GraphXfer : std::enable_shared_from_this<GraphXfer> {
public:
  GraphXfer(std::shared_ptr<Model> _model);
  static void load_graph_xfer_from_pb_file(std::shared_ptr<Model> model,
                                           std::vector<std::shared_ptr<GraphXfer>>& xfers,
                                           std::string filename);
  TensorX new_tensor(void);
  bool can_match(std::shared_ptr<OpX> srcOp, Op op, std::shared_ptr<Graph> graph);
  void match(std::shared_ptr<OpX> srcOp, Op op, std::shared_ptr<Graph> graph);
  void unmatch(std::shared_ptr<OpX> srcOp, Op op, std::shared_ptr<Graph> graph);
  void create_operator_from_pb(const GraphSubst::Operator& pbOp,
                               std::map<int, TensorX>& mappedInputs,
                               bool isSrcOp = true);
  std::shared_ptr<OpX> create_activation(TensorX input, OpType type, bool isSrcOp = true);
  std::shared_ptr<OpX> create_conv2d(TensorX input, TensorX weight,
                     //int kernelH, int kernelW,
                     int strideH, int strideW,
                     PaddingMode padding,
                     ActiMode activation,
                     bool isSrcOp = true);
  std::shared_ptr<OpX> create_batchnorm(TensorX input, TensorX scale, TensorX bias,
                        TensorX mean, TensorX var, bool isSrcOp = true);
  std::shared_ptr<OpX> create_element(TensorX input0, TensorX input1,
                      OpType type, bool isSrcOp = true);
  std::shared_ptr<OpX> create_fuse_conv_batchnorm(TensorX conv_w, TensorX scale,
                                  TensorX bias, TensorX mean, TensorX var,
                                  bool isSrcOp = true);
  std::shared_ptr<OpX> create_fuse_conv_batchnorm_alpha_var(TensorX conv_w, TensorX scale, 
                                            TensorX var, bool isSrcOp = true);
  std::shared_ptr<OpX> create_fuse_conv_batchnorm_bias(TensorX scale,
                                           TensorX bias, TensorX mean,
                                           TensorX var, bool isSrcOp = true);
  std::shared_ptr<OpX> create_broadcast_add(TensorX data, TensorX bias, bool isSrcOp = true);
  std::shared_ptr<OpX> create_pool2d_avg(TensorX input, TensorX weight,
                         //int kernelH, int kernelW,
                         int strideH, int strideW,
                         PaddingMode padding,
                         ActiMode activation,
                         bool isSrcOp = true);
  std::shared_ptr<OpX> create_matmul(TensorX input, TensorX weight,
                     ActiMode activation, bool isSrcOp = true);
  std::shared_ptr<OpX> create_mul(TensorX x, TensorX y, bool isSrcOp = true);
  std::shared_ptr<OpX> create_transpose(TensorX input, int numDim, int* perm, int shuffle);
  std::shared_ptr<OpX> create_enlarge(TensorX w1, TensorX w2, bool isSrcOp = true);
  std::shared_ptr<OpX> create_merge_gconv(TensorX w, int count, bool isSrcOp = true);
  std::shared_ptr<OpX> create_concat(int axis, int numDim, TensorX in1, TensorX in2, bool isSrcOp = true);
  std::shared_ptr<OpX> create_concat(int axis, int numDim, int n, TensorX* ins, bool isSrcOp = true);
  std::shared_ptr<OpX> create_split(TensorX input, int axis, int n, bool isSrcOp = true);
  void add_src_op(std::shared_ptr<SrcOp> op);
  void add_dst_op(std::shared_ptr<DstOp> op);
  void add_src_edge(std::shared_ptr<SrcOp> src, std::shared_ptr<SrcOp> tgt, int srcIdx = 0, int dstIdx = 0);
  void add_dst_edge(std::shared_ptr<DstOp> src, std::shared_ptr<DstOp> tgt, int srcIdx = 0, int dstIdx = 0);
  bool add_constraint(Compare comp, std::shared_ptr<SrcOp> src, PMParameter srcPara,
                      std::shared_ptr<SrcOp> tgt, PMParameter dstPara);
  bool map_input(std::shared_ptr<SrcOp> src, std::shared_ptr<DstOp> dst);
  bool map_output(std::shared_ptr<SrcOp> src, std::shared_ptr<DstOp> dst);
  bool map_output(TensorX src, TensorX dst);
  void run(int depth, std::shared_ptr<Graph> graph,
           std::priority_queue<std::shared_ptr<Graph>, std::vector<std::shared_ptr<Graph>>, GraphCompare>&,
           std::set<size_t>&, float threshold, int maxNumOps);
  std::shared_ptr<Graph> create_new_graph(std::shared_ptr<Graph> graph);
  bool create_new_operator(const std::shared_ptr<OpX> opx, Op& op);

  // built-in substitutions
  static std::shared_ptr<GraphXfer> create_conv_relu(std::shared_ptr<Model> model, int strideH, int strideW, PaddingMode padding);
  static std::shared_ptr<GraphXfer> create_conv_batch(std::shared_ptr<Model> model, int strideH, int strideW, PaddingMode padding);
  static std::shared_ptr<GraphXfer> create_conv_mul(std::shared_ptr<Model> model, int strideH, int strideW, PaddingMode padding);
  static std::shared_ptr<GraphXfer> create_conv_add(std::shared_ptr<Model> model, int strideH, int strideW, PaddingMode padding);
  static std::shared_ptr<GraphXfer> create_enlarge_merge_convs(std::shared_ptr<Model> model, ActiMode activation);
  static std::shared_ptr<GraphXfer> create_merge_group_convs(std::shared_ptr<Model> model, int strideH, int strideW, ActiMode activation);
public:
  std::shared_ptr<Model> model;
  int tensorId;
  //std::vector<TwoOpConstraint> constraints;
  //std::map<std::shared_ptr<SrcOp>, std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> > > srcInEdges, srcOutEdges;
  //std::map<std::shared_ptr<DstOp>, std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > > dstInEdges, dstOutEdges;
  std::map<Op, std::shared_ptr<OpX>, OpCompare> mappedOps;
  std::multimap<int, std::pair<Op, int> > mappedInputs;
  std::map<TensorX, TensorX, TensorXCompare> mappedOutputs;
  std::vector<std::shared_ptr<OpX>> srcOps;
  std::vector<std::shared_ptr<OpX>> dstOps;
};

} // namespace XFlow
#endif
