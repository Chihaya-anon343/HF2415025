#pragma once
#include "common/Types.hpp"
#include <ostream>
#include <iomanip>

namespace gpnp {
inline void dualCornerRecoveryHeader(std::ostream& out) {
    out<<"frame,strategy,pose_success,class_id,stage,recovery_status,id,original_x,original_y,"
          "predicted_x,predicted_y,final_x,final_y,residual,threshold_px,action,used_in_pose\n"<<std::setprecision(17);
}
inline void writeDualCornerRecovery(std::ostream& out,int frame,const PipelineResult& result) {
    for(const auto& record:result.dual_corner_observations) {
        const auto& o=record.point;
        out<<frame<<','<<result.strategy_name<<','<<result.success<<','<<record.class_id<<','
           <<record.stage<<','<<record.recovery_status<<','<<o.id<<',';
        if(o.status!="not_observed")out<<o.original.x<<','<<o.original.y<<',';
        else out<<",,";
        if(o.has_prediction)out<<o.predicted.x<<','<<o.predicted.y;
        else out<<',';
        out<<',';
        if(o.status!="not_observed")out<<o.final_point.x<<','<<o.final_point.y;
        else out<<',';
        out<<','<<o.residual<<','<<o.threshold_px<<','<<o.status<<','<<record.used_in_pose<<'\n';
    }
}
} // namespace gpnp
