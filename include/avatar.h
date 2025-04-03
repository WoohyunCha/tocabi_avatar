#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <std_msgs/String.h>

#include "math_type_define.h"
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include "tocabi_msgs/matrix_3_4.h"
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/String.h>
#include <sstream>
#include <fstream>
//#include "tocabi_msgs/FTsensor.h" // real robot experiment

//lexls
// #include <lexls/lexlsi.h>
// #include <lexls/tools.h>
// #include <lexls>

#include <iomanip>
#include <iostream>

// pedal
#include <ros/ros.h>
#include "tocabi_msgs/WalkingCommand.h"
#include <std_msgs/Float32.h>

#include <eigen_conversions/eigen_msg.h>

#include <random>
#include <cmath>
#include <sensor_msgs/Joy.h>

const int FILE_CNT = 2;

// mob lstm
const int n_input_ = 24;
const int n_sequence_length_ = 1;
const int n_output_ = 12;
const int n_hidden_ = 128;
const int buffer_size_ = n_input_ * n_sequence_length_ * 20;
const int nn_input_size_ = n_input_ * n_sequence_length_;
const bool gaussian_mode_ = true;

const std::string FILE_NAMES[FILE_CNT] =
{
  ///change this directory when you use this code on the other computer///
    "/home/dyros/data/dg/training_data.txt",
    "/home/dyros/data/dg/hand_ft.txt"
    // "/home/dyros/data/dg/1_com_.txt",
    // "/home/dyros/data/dg/2_zmp_.txt",
    // "/home/dyros/data/dg/3_foot_.txt",
    // "/home/dyros/data/dg/4_torque_.txt",
    // "/home/dyros/data/dg/5_joint_.txt",
    // "/home/dyros/data/dg/6_hand_.txt",
    // "/home/dyros/data/dg/7_elbow_.txt",
    // "/home/dyros/data/dg/8_shoulder_.txt",
    // "/home/dyros/data/dg/9_acromion_.txt",
    // "/home/dyros/data/dg/10_hmd_.txt",
    // "/home/dyros/data/dg/11_tracker_.txt",
    // "/home/dyros/data/dg/12_qpik_.txt",
    // "/home/dyros/data/dg/13_tracker_vel_.txt"
};

// const std::string calibration_folder_dir_ = "/home/dyros/data/vive_tracker/calibration_log/dh";  //tocabi 
const std::string calibration_folder_dir_ = "/home/yong20/Downloads";    //yong pc

class AvatarController
{
public:
    AvatarController(RobotData &rd);
    // ~AvatarController();
    Eigen::VectorQd getControl();
    std::ofstream file[FILE_CNT];
    std::ofstream calibration_log_file_ofstream_[4];
    std::ifstream calibration_log_file_ifstream_[4];
    //void taskCommandToCC(TaskCommand tc_);

    void computeSlow();
    void computeFast();
    void computeThread3();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);


    RobotData &rd_;
    RobotData rd_cc_;

    ros::NodeHandle nh_avatar_;
    ros::CallbackQueue queue_avatar_;

    std::vector<CQuadraticProgram> QP_qdot_hqpik_;        
    CQuadraticProgram QP_motion_retargeting_[3];    // task1: each arm, task2: relative arm, task3: hqp second hierarchy

    std::atomic<bool> atb_desired_q_update_{false};

    RigidBodyDynamics::Model model_d_;  //updated by desired q
    RigidBodyDynamics::Model model_C_;  //for calcuating Coriolis matrix
    RigidBodyDynamics::Model model_global_;  //state manager coordinate
    RigidBodyDynamics::Model model_local_;  //local pelvis position gravity orientation
    RigidBodyDynamics::Model model_MJ_;  //for calcuating CMM

    VectorVQd __q_dot_virtual;
    VectorQVQd __q_virtual;
    VectorVQd __q_ddot_virtual;
    LinkData link_avatar_[LINK_NUMBER + 1];


    //////////dg custom controller functions////////
    void setGains();
    void getRobotData();
    void getProcessedRobotData();
    void motionGenerator();

    //motion control
    void motionRetargeting_HQPIK();
    // void motionRetargeting_HQPIK_lexls();
    void rawMasterPoseProcessing();
    void handPositionRetargeting();
    void orientationRetargeting();
    void poseCalibration();
    void getCenterOfShoulderCali(Eigen::Vector3d Still_pose_cali, Eigen::Vector3d T_pose_cali, Eigen::Vector3d Forward_pose_cali, Eigen::Vector3d &CenterOfShoulder_cali);
    
    void qpRetargeting_1();
    
    void getTranslationDataFromText(std::ifstream &text_file, Eigen::Vector3d &trans);
    void getMatrix3dDataFromText(std::ifstream &text_file, Eigen::Matrix3d &mat);
    void getIsometry3dDataFromText(std::ifstream &text_file, Eigen::Isometry3d &isom);

    void savePreData();
    void printOutTextFile();

    /////////////////////////////////////////////////////////

    //////////dg ROS related////////
    ros::Subscriber upperbodymode_sub;

    ros::Subscriber arm_pd_gain_sub;
    ros::Subscriber waist_pd_gain_sub;

    // master sensor
    ros::Subscriber hmd_posture_sub;
    ros::Subscriber left_controller_posture_sub;
    ros::Subscriber right_controller_posture_sub;
    
    ros::Subscriber lhand_tracker_posture_sub;
    ros::Subscriber rhand_tracker_posture_sub;
    ros::Subscriber lelbow_tracker_posture_sub;
    ros::Subscriber relbow_tracker_posture_sub;
    ros::Subscriber chest_tracker_posture_sub;
    ros::Subscriber pelvis_tracker_posture_sub;

    ros::Subscriber tracker_status_sub;
    ros::Subscriber tracker_pose_sub;
    ros::Subscriber master_pose_sub;

    ros::Subscriber vive_tracker_pose_calibration_sub;
    ros::Subscriber robot_hand_pos_mapping_scale_sub;

    ros::Publisher calibration_state_pub;
    ros::Publisher calibration_state_gui_log_pub;

    ros::Publisher upperbodymode_pub;
    ros::Publisher avatar_warning_pub;

    ros::Publisher haptic_force_pub;

    void UpperbodyModeCallback(const std_msgs::Int8 &msg);

    void ArmJointGainCallback(const std_msgs::Float32MultiArray &msg);
    void WaistJointGainCallback(const std_msgs::Float32MultiArray &msg);

    // HMD + Tracker related
    void HmdCallback(const tocabi_msgs::matrix_3_4 &msg);
    void PoseCalibrationCallback(const std_msgs::Int8 &msg);
    void TrackerStatusCallback(const std_msgs::Bool &msg);

    void TrackerPoseCallback(const geometry_msgs::PoseArray &msg);
    void MasterPoseCallback(const geometry_msgs::PoseArray &msg);

    void HandPosMappingScaleCallback(const std_msgs::Float32 &msg);

    ///////////////////////////////

    ////////////////dg custom controller variables/////////////
    /////
    ///// global: variables represented from the gravity alined frame which is attatched at the pelvis frame
    ///// local: variables represented from its link frame
    /////
    ///////////////////////////////////////////////////////////

    int upper_body_mode_;
    int upper_body_mode_raw_;                               // 1: init pose,  2: zero pose, 3: swing arm 4: motion retarggeting
    bool walking_mode_on_;                                  // turns on when the walking control command is received and truns off after saving start time
    bool chair_mode_;                                       // For chair sitting mode
    bool float_data_collect_mode_ = false;                  // For data collection in the air

    double program_start_time_;
    
    double current_time_;
    double pre_time_;
    double start_time_;
    double dt_;
    double init_leg_time_;  //for first smothing of leg joint angle
    
    double com_mass_;

    int mode_12_count_ = 0;
    int setcontact_flag = 0;

    Eigen::VectorQd torque_task_max_;
    Eigen::VectorQd torque_task_min_;

    Eigen::VectorQd torque_upper_fast_;
    Eigen::VectorQd torque_upper_;
    Eigen::VectorQd torque_lower_;

    // Pevlis related variables
    Eigen::Vector3d pelv_pos_current_;
    Eigen::Vector6d pelv_vel_current_;
    Eigen::Vector3d pelv_angvel_current_;
    Eigen::Matrix3d pelv_rot_current_;
    Eigen::Vector3d pelv_rpy_current_;
    Eigen::Matrix3d pelv_rot_current_yaw_aline_;
    Eigen::Matrix3d pelv_yaw_rot_current_from_global_;
    Eigen::Isometry3d pelv_transform_current_from_global_;
    double pelv_height_offset_;
     
    Eigen::Vector3d pelv_pos_init_;
    Eigen::Vector6d pelv_vel_init_;
    Eigen::Matrix3d pelv_rot_init_;
    Eigen::Vector3d pelv_rpy_init_;
    Eigen::Matrix3d pelv_rot_init_yaw_aline_;
    Eigen::Isometry3d pelv_transform_init_from_global_;
    Eigen::Isometry3d pelv_trajectory_support_init_;


    // Joint related variables
    Eigen::VectorQd current_q_;
    Eigen::VectorQd current_q_dot_;
    Eigen::VectorQd current_q_ddot_;
    Eigen::VectorQd desired_q_;
    Eigen::VectorQd desired_q_dot_;
    Eigen::VectorQd desired_q_ddot_;
    Eigen::VectorQd pre_q_;
    Eigen::VectorQd pre_desired_q_;
    Eigen::VectorQd pre_desired_q_dot_;
    Eigen::VectorQd last_desired_q_;
    Eigen::VectorQVQd pre_desired_q_qvqd_;
    Eigen::VectorVQd pre_desired_q_dot_vqd_;
    Eigen::VectorVQd pre_desired_q_ddot_vqd_;

    Eigen::VectorXd q_ddot_virtual_Xd_global_, q_dot_virtual_Xd_global_, q_dot_virtual_Xd_global_pre_, q_virtual_Xd_global_; // for model_global_ update
    Eigen::VectorXd q_ddot_virtual_Xd_local_, q_dot_virtual_Xd_local_, q_dot_virtual_Xd_local_pre_, q_virtual_Xd_local_;

    Eigen::VectorXd q_ddot_virtual_Xd_global_noise_, q_dot_virtual_Xd_global_noise_, q_virtual_Xd_global_noise_;

    Eigen::VectorQd desired_q_fast_;
    Eigen::VectorQd desired_q_dot_fast_;
    Eigen::VectorQd desired_q_slow_;
    Eigen::VectorQd desired_q_dot_slow_;

    //sca
    Eigen::VectorQd q_ddot_max_slow_;
    Eigen::VectorQd q_ddot_max_fast_;
    Eigen::VectorQd q_ddot_max_thread_;
    Eigen::VectorQd q_braking_stop_;
    Eigen::VectorQd torque_max_braking_;

    Eigen::VectorQd motion_q_;
    Eigen::VectorQd motion_q_dot_;
    Eigen::VectorQd motion_q_pre_;
    Eigen::VectorQd motion_q_dot_pre_;
    Eigen::VectorQd init_q_;
    Eigen::VectorQd zero_q_;
    Eigen::VectorQVQd init_q_virtual_;

    Eigen::VectorQd kp_joint_;
	Eigen::VectorQd kv_joint_;

	Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_stiff_joint_;
	Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_stiff_joint_;
    // walking controller variables
    Eigen::VectorQd pd_control_mask_; //1 for joint ik pd control

    //getLegIK
    Eigen::Isometry3d lfoot_transform_desired_;
	Eigen::Isometry3d rfoot_transform_desired_;
	Eigen::Isometry3d pelv_transform_desired_;
    Eigen::Isometry3d lfoot_transform_desired_last_;
	Eigen::Isometry3d rfoot_transform_desired_last_;
	Eigen::Isometry3d pelv_transform_desired_last_;

    Eigen::MatrixXd jac_com_;
    Eigen::MatrixXd jac_com_pos_;
    
    Eigen::Isometry3d pelv_transform_start_from_global_;
    Eigen::Isometry3d rfoot_transform_start_from_global_;
    Eigen::Isometry3d lfoot_transform_start_from_global_;

    Eigen::Isometry3d upperbody_transform_init_from_global_;
    Eigen::Isometry3d head_transform_init_from_global_;
    Eigen::Isometry3d lfoot_transform_init_from_global_;
    Eigen::Isometry3d rfoot_transform_init_from_global_;
    Eigen::Isometry3d lhand_transform_init_from_global_;
    Eigen::Isometry3d rhand_transform_init_from_global_;
    Eigen::Isometry3d lelbow_transform_init_from_global_;
    Eigen::Isometry3d relbow_transform_init_from_global_;
    Eigen::Isometry3d lupperarm_transform_init_from_global_; //4rd axis of arm joint 
    Eigen::Isometry3d rupperarm_transform_init_from_global_; 
    Eigen::Isometry3d lshoulder_transform_init_from_global_; //3rd axis of arm joint 
    Eigen::Isometry3d rshoulder_transform_init_from_global_; 
    Eigen::Isometry3d lacromion_transform_init_from_global_; //2nd axis of arm joint (견봉)
    Eigen::Isometry3d racromion_transform_init_from_global_;
    Eigen::Isometry3d larmbase_transform_init_from_global_; //1st axis of arm joint (견봉)
    Eigen::Isometry3d rarmbase_transform_init_from_global_;

    Eigen::Isometry3d upperbody_transform_current_from_global_;
    Eigen::Isometry3d head_transform_current_from_global_;
    Eigen::Isometry3d lfoot_transform_current_from_global_;
    Eigen::Isometry3d rfoot_transform_current_from_global_;
    Eigen::Isometry3d lhand_transform_current_from_global_;
    Eigen::Isometry3d rhand_transform_current_from_global_;
    Eigen::Isometry3d lelbow_transform_current_from_global_;
    Eigen::Isometry3d relbow_transform_current_from_global_;
    Eigen::Isometry3d lupperarm_transform_current_from_global_; //4th axis of arm joint
    Eigen::Isometry3d rupperarm_transform_current_from_global_;
    Eigen::Isometry3d lshoulder_transform_current_from_global_; //3rd axis of arm joint
    Eigen::Isometry3d rshoulder_transform_current_from_global_;
    Eigen::Isometry3d lacromion_transform_current_from_global_; //2nd axis of arm joint (견봉)
    Eigen::Isometry3d racromion_transform_current_from_global_;
    Eigen::Isometry3d larmbase_transform_current_from_global_; //1st axis of arm joint 
    Eigen::Isometry3d rarmbase_transform_current_from_global_;

    Eigen::Isometry3d lknee_transform_current_from_global_;
    Eigen::Isometry3d rknee_transform_current_from_global_;

    Eigen::Vector3d lhand_rpy_current_from_global_;
    Eigen::Vector3d rhand_rpy_current_from_global_;
    Eigen::Vector3d lelbow_rpy_current_from_global_;
    Eigen::Vector3d relbow_rpy_current_from_global_;
    Eigen::Vector3d lupperarm_rpy_current_from_global_;
    Eigen::Vector3d rupperarm_rpy_current_from_global_;
    Eigen::Vector3d lshoulder_rpy_current_from_global_;
    Eigen::Vector3d rshoulder_rpy_current_from_global_;
    Eigen::Vector3d lacromion_rpy_current_from_global_;
    Eigen::Vector3d racromion_rpy_current_from_global_;

    Eigen::Isometry3d upperbody_transform_pre_desired_from_;
    Eigen::Isometry3d head_transform_pre_desired_from_;
    Eigen::Isometry3d lfoot_transform_pre_desired_from_;
    Eigen::Isometry3d rfoot_transform_pre_desired_from_;
    Eigen::Isometry3d lhand_transform_pre_desired_from_;
    Eigen::Isometry3d rhand_transform_pre_desired_from_;
    Eigen::Isometry3d lelbow_transform_pre_desired_from_;
    Eigen::Isometry3d relbow_transform_pre_desired_from_;
    Eigen::Isometry3d lupperarm_transform_pre_desired_from_;
    Eigen::Isometry3d rupperarm_transform_pre_desired_from_;
    Eigen::Isometry3d lshoulder_transform_pre_desired_from_;
    Eigen::Isometry3d rshoulder_transform_pre_desired_from_;
    Eigen::Isometry3d lacromion_transform_pre_desired_from_;
    Eigen::Isometry3d racromion_transform_pre_desired_from_;
    Eigen::Isometry3d larmbase_transform_pre_desired_from_; //1st axis of arm joint
    Eigen::Isometry3d rarmbase_transform_pre_desired_from_;

    Eigen::Vector3d lhand_control_point_offset_, rhand_control_point_offset_;   //red_hand made by sy

    Eigen::Vector6d lfoot_vel_current_from_global_;
    Eigen::Vector6d rfoot_vel_current_from_global_;
    Eigen::Vector6d lhand_vel_current_from_global_;
    Eigen::Vector6d rhand_vel_current_from_global_;
    Eigen::Vector6d lelbow_vel_current_from_global_;
    Eigen::Vector6d relbow_vel_current_from_global_;
    Eigen::Vector6d lupperarm_vel_current_from_global_;
    Eigen::Vector6d rupperarm_vel_current_from_global_;
    Eigen::Vector6d lshoulder_vel_current_from_global_;
    Eigen::Vector6d rshoulder_vel_current_from_global_;
    Eigen::Vector6d lacromion_vel_current_from_global_;
    Eigen::Vector6d racromion_vel_current_from_global_;

    Eigen::Isometry3d lfoot_transform_init_from_support_;
    Eigen::Isometry3d rfoot_transform_init_from_support_;
    Eigen::Isometry3d pelv_transform_init_from_support_;
    Eigen::Vector3d pelv_rpy_init_from_support_;

    Eigen::Isometry3d lfoot_transform_start_from_support_;
    Eigen::Isometry3d rfoot_transform_start_from_support_;
    Eigen::Isometry3d pelv_transform_start_from_support_;

    Eigen::Isometry3d lfoot_transform_current_from_support_;
    Eigen::Isometry3d rfoot_transform_current_from_support_;
    Eigen::Isometry3d pelv_transform_current_from_support_;

    Eigen::Vector6d lh_ft_feedback_;
    Eigen::Vector6d rh_ft_feedback_;

    //MotionRetargeting variables
    int upperbody_mode_recieved_;
    double upperbody_command_time_;
    Eigen::VectorQd upperbody_mode_q_init_;

    Eigen::Isometry3d master_lhand_pose_raw_;
    Eigen::Isometry3d master_rhand_pose_raw_;
    Eigen::Isometry3d master_head_pose_raw_;
    Eigen::Isometry3d master_lelbow_pose_raw_;
    Eigen::Isometry3d master_relbow_pose_raw_;
    Eigen::Isometry3d master_lshoulder_pose_raw_;
    Eigen::Isometry3d master_rshoulder_pose_raw_;
    Eigen::Isometry3d master_upperbody_pose_raw_;

    Eigen::Isometry3d master_lhand_pose_raw_pre_;
    Eigen::Isometry3d master_rhand_pose_raw_pre_;
    Eigen::Isometry3d master_head_pose_raw_pre_;
    Eigen::Isometry3d master_lelbow_pose_raw_pre_;
    Eigen::Isometry3d master_relbow_pose_raw_pre_;
    Eigen::Isometry3d master_lshoulder_pose_raw_pre_;
    Eigen::Isometry3d master_rshoulder_pose_raw_pre_;
    Eigen::Isometry3d master_upperbody_pose_raw_pre_;

    Eigen::Isometry3d master_lhand_pose_raw_ppre_;
    Eigen::Isometry3d master_rhand_pose_raw_ppre_;
    Eigen::Isometry3d master_head_pose_raw_ppre_;
    Eigen::Isometry3d master_lelbow_pose_raw_ppre_;
    Eigen::Isometry3d master_relbow_pose_raw_ppre_;
    Eigen::Isometry3d master_lshoulder_pose_raw_ppre_;
    Eigen::Isometry3d master_rshoulder_pose_raw_ppre_;
    Eigen::Isometry3d master_upperbody_pose_raw_ppre_;

    Eigen::Isometry3d master_lhand_pose_;
    Eigen::Isometry3d master_rhand_pose_;
    Eigen::Isometry3d master_head_pose_;
    Eigen::Isometry3d master_lelbow_pose_;
    Eigen::Isometry3d master_relbow_pose_;
    Eigen::Isometry3d master_lshoulder_pose_;
    Eigen::Isometry3d master_rshoulder_pose_;
    Eigen::Isometry3d master_upperbody_pose_;

    Eigen::Isometry3d master_lhand_pose_pre_;
    Eigen::Isometry3d master_rhand_pose_pre_;
    Eigen::Isometry3d master_head_pose_pre_;
    Eigen::Isometry3d master_lelbow_pose_pre_;
    Eigen::Isometry3d master_relbow_pose_pre_;
    Eigen::Isometry3d master_lshoulder_pose_pre_;
    Eigen::Isometry3d master_rshoulder_pose_pre_;
    Eigen::Isometry3d master_upperbody_pose_pre_;

    Eigen::Isometry3d master_lhand_pose_ppre_;
    Eigen::Isometry3d master_rhand_pose_ppre_;
    Eigen::Isometry3d master_head_pose_ppre_;
    Eigen::Isometry3d master_lelbow_pose_ppre_;
    Eigen::Isometry3d master_relbow_pose_ppre_;
    Eigen::Isometry3d master_lshoulder_pose_ppre_;
    Eigen::Isometry3d master_rshoulder_pose_ppre_;
    Eigen::Isometry3d master_upperbody_pose_ppre_;

    Eigen::Vector6d master_lhand_vel_;
    Eigen::Vector6d master_rhand_vel_;
    Eigen::Vector6d master_head_vel_;
    Eigen::Vector6d master_lelbow_vel_;
    Eigen::Vector6d master_relbow_vel_;
    Eigen::Vector6d master_lshoulder_vel_;
    Eigen::Vector6d master_rshoulder_vel_;
    Eigen::Vector6d master_upperbody_vel_;

    Eigen::Vector3d master_lhand_rqy_;
    Eigen::Vector3d master_rhand_rqy_;
    Eigen::Vector3d master_lelbow_rqy_;
    Eigen::Vector3d master_relbow_rqy_;
    Eigen::Vector3d master_lshoulder_rqy_;
    Eigen::Vector3d master_rshoulder_rqy_;
    
    Eigen::Isometry3d master_lhand_pose_start_;
    Eigen::Isometry3d master_rhand_pose_start_;

    Eigen::Vector3d master_head_rqy_;

    Eigen::Vector3d master_relative_lhand_pos_raw_;
    Eigen::Vector3d master_relative_rhand_pos_raw_;
    Eigen::Vector3d master_relative_lhand_pos_;
    Eigen::Vector3d master_relative_rhand_pos_;
    Eigen::Vector3d master_relative_lhand_pos_pre_;
    Eigen::Vector3d master_relative_rhand_pos_pre_;

    double robot_arm_max_l_;
    double robot_upperarm_max_l_;
    double robot_lowerarm_max_l_;
    double robot_shoulder_width_;

    double hand_pos_mapping_scale_raw_ = 1.0;
    ////////////HMD + VIVE TRACKER////////////

    bool hmd_tracker_status_raw_;   //1: good, 0: bad
    bool hmd_tracker_status_;   //1: good, 0: bad
    bool hmd_tracker_status_pre_;   //1: good, 0: bad

    double tracker_status_changed_time_;
    
    bool master_arm_mode_ = false;
    bool real_robot_mode_ = false;

    double hmd_larm_max_l_;
    double hmd_rarm_max_l_;
    double hmd_shoulder_width_;

    bool hmd_check_pose_calibration_[5];   // 0: still cali, 1: T pose cali, 2: Forward Stretch cali, 3: Calibration is completed, 4: read calibration from log file
    bool still_pose_cali_flag_;
    bool t_pose_cali_flag_;
    bool forward_pose_cali_flag_;
    bool read_cali_log_flag_;

    Eigen::Isometry3d hmd_head_pose_raw_;
    Eigen::Isometry3d hmd_lshoulder_pose_raw_;
    Eigen::Isometry3d hmd_lupperarm_pose_raw_;
    Eigen::Isometry3d hmd_lhand_pose_raw_;
    Eigen::Isometry3d hmd_rshoulder_pose_raw_;
    Eigen::Isometry3d hmd_rupperarm_pose_raw_;
    Eigen::Isometry3d hmd_rhand_pose_raw_;
    Eigen::Isometry3d hmd_chest_pose_raw_;
    Eigen::Isometry3d hmd_pelv_pose_raw_;

    Eigen::Isometry3d hmd_head_pose_raw_last_;
    Eigen::Isometry3d hmd_lshoulder_pose_raw_last_;
    Eigen::Isometry3d hmd_lupperarm_pose_raw_last_;
    Eigen::Isometry3d hmd_lhand_pose_raw_last_;
    Eigen::Isometry3d hmd_rshoulder_pose_raw_last_;
    Eigen::Isometry3d hmd_rupperarm_pose_raw_last_;
    Eigen::Isometry3d hmd_rhand_pose_raw_last_;
    Eigen::Isometry3d hmd_chest_pose_raw_last_;
    Eigen::Isometry3d hmd_pelv_pose_raw_last_;

    Eigen::Isometry3d hmd_head_pose_;
    Eigen::Isometry3d hmd_lshoulder_pose_;
    Eigen::Isometry3d hmd_lupperarm_pose_;
    Eigen::Isometry3d hmd_lhand_pose_;
    Eigen::Isometry3d hmd_rshoulder_pose_;
    Eigen::Isometry3d hmd_rupperarm_pose_;
    Eigen::Isometry3d hmd_rhand_pose_;
    Eigen::Isometry3d hmd_chest_pose_;
    Eigen::Isometry3d hmd_pelv_pose_;

    Eigen::Isometry3d hmd_head_pose_pre_;
    Eigen::Isometry3d hmd_lshoulder_pose_pre_;
    Eigen::Isometry3d hmd_lupperarm_pose_pre_;
    Eigen::Isometry3d hmd_lhand_pose_pre_;
    Eigen::Isometry3d hmd_rshoulder_pose_pre_;
    Eigen::Isometry3d hmd_rupperarm_pose_pre_;
    Eigen::Isometry3d hmd_rhand_pose_pre_;
    Eigen::Isometry3d hmd_chest_pose_pre_;
    Eigen::Isometry3d hmd_pelv_pose_pre_;

    Eigen::Isometry3d hmd_head_pose_init_;  
    Eigen::Isometry3d hmd_lshoulder_pose_init_; 
    Eigen::Isometry3d hmd_lupperarm_pose_init_;
    Eigen::Isometry3d hmd_lhand_pose_init_;
    Eigen::Isometry3d hmd_rshoulder_pose_init_; 
    Eigen::Isometry3d hmd_rupperarm_pose_init_;
    Eigen::Isometry3d hmd_rhand_pose_init_;
    Eigen::Isometry3d hmd_chest_pose_init_;
    Eigen::Isometry3d hmd_pelv_pose_init_;

    Eigen::Isometry3d hmd_lhand_pose_start_;
    Eigen::Isometry3d hmd_rhand_pose_start_;

    Eigen::Vector3d hmd2robot_lhand_pos_mapping_;
	Eigen::Vector3d hmd2robot_rhand_pos_mapping_;
    Eigen::Vector3d hmd2robot_lelbow_pos_mapping_;
    Eigen::Vector3d hmd2robot_relbow_pos_mapping_;

    Eigen::Vector3d hmd2robot_lhand_pos_mapping_init_;
    Eigen::Vector3d hmd2robot_rhand_pos_mapping_init_;

    Eigen::Vector3d hmd_still_cali_lhand_pos_;
    Eigen::Vector3d hmd_still_cali_rhand_pos_;
    Eigen::Vector3d hmd_tpose_cali_lhand_pos_;
    Eigen::Vector3d hmd_tpose_cali_rhand_pos_;
    Eigen::Vector3d hmd_forward_cali_lhand_pos_;
    Eigen::Vector3d hmd_forward_cali_rhand_pos_;
    
    Eigen::Vector3d hmd_lshoulder_center_pos_;
    Eigen::Vector3d hmd_rshoulder_center_pos_;

    Eigen::Vector3d hmd_chest_2_lshoulder_center_pos_;
    Eigen::Vector3d hmd_chest_2_rshoulder_center_pos_;

    //global frame of vive tracker
    Eigen::Vector6d hmd_head_vel_global_;
    Eigen::Vector6d hmd_lupperarm_vel_global_;
    Eigen::Vector6d hmd_lhand_vel_global_;
    Eigen::Vector6d hmd_rupperarm_vel_global_;
    Eigen::Vector6d hmd_rhand_vel_global_;
    Eigen::Vector6d hmd_chest_vel_global_;
    Eigen::Vector6d hmd_pelv_vel_global_; 

    //pelvis frame
    Eigen::Vector6d hmd_head_vel_;
    Eigen::Vector6d hmd_lshoulder_vel_;
    Eigen::Vector6d hmd_lupperarm_vel_;
    Eigen::Vector6d hmd_lhand_vel_;
    Eigen::Vector6d hmd_rshoulder_vel_;
    Eigen::Vector6d hmd_rupperarm_vel_;
    Eigen::Vector6d hmd_rhand_vel_;
    Eigen::Vector6d hmd_chest_vel_;
    Eigen::Vector6d hmd_pelv_vel_; 
    
    int hmd_head_abrupt_motion_count_;
    int hmd_lupperarm_abrupt_motion_count_;
    int hmd_lhand_abrupt_motion_count_;
    int hmd_rupperarm_abrupt_motion_count_;
    int hmd_rhand_abrupt_motion_count_;
    int hmd_chest_abrupt_motion_count_;
    int hmd_pelv_abrupt_motion_count_;

    ///////////QPIK///////////////////////////
    Eigen::Vector3d lhand_pos_error_;
    Eigen::Vector3d rhand_pos_error_;
    Eigen::Vector3d lhand_ori_error_;
    Eigen::Vector3d rhand_ori_error_;

    Eigen::Vector3d lelbow_ori_error_;
    Eigen::Vector3d relbow_ori_error_;
    Eigen::Vector3d lshoulder_ori_error_;
    Eigen::Vector3d rshoulder_ori_error_;

    Eigen::Vector6d lhand_vel_error_;
    Eigen::Vector6d rhand_vel_error_;
    Eigen::Vector3d lelbow_vel_error_;
    Eigen::Vector3d relbow_vel_error_;
    Eigen::Vector3d lacromion_vel_error_;
    Eigen::Vector3d racromion_vel_error_;
    //////////////////////////////////////////

    /////////////QPIK UPPERBODY /////////////////
    const int hierarchy_num_upperbody_ = 4;
    const int variable_size_upperbody_ = 21;
	const int constraint_size1_upperbody_ = 21;	//[lb <=	x	<= 	ub] form constraints
	const int constraint_size2_upperbody_ = 12;	//[lb <=	Ax 	<=	ub] from constraints
   	const int control_size_upperbody_[4] = {3, 14, 4, 4};		//1: upperbody, 2: head + hand, 3: upperarm, 4: shoulder

	// const int control_size_hand = 12;		//2
	// const int control_size_upperbody = 3;	//1
	// const int control_size_head = 2;		//2
	// const int control_size_upperarm = 4; 	//3
	// const int control_size_shoulder = 4;	//4

    double w1_upperbody_;
    double w2_upperbody_;
    double w3_upperbody_;
    double w4_upperbody_;
    double w5_upperbody_;
    double w6_upperbody_;

    Eigen::MatrixXd H_upperbody_, A_upperbody_;
    Eigen::MatrixXd J_upperbody_[4];
    Eigen::VectorXd g_upperbody_, u_dot_upperbody_[4], qpres_upperbody_, ub_upperbody_, lb_upperbody_, ubA_upperbody_, lbA_upperbody_;
    Eigen::VectorXd q_dot_upperbody_;

    Eigen::MatrixXd N1_upperbody_, N2_aug_upperbody_, N3_aug_upperbody_;
    Eigen::MatrixXd J2_aug_upperbody_, J2_aug_pinv_upperbody_, J3_aug_upperbody_, J3_aug_pinv_upperbody_, J1_pinv_upperbody_,  J2N1_upperbody_, J3N2_aug_upperbody_, J4N3_aug_upperbody_;
    Eigen::MatrixXd I3_upperbody_, I6_upperbody_, I15_upperbody_, I21_upperbody_;
    /////////////////////////////////////////////

    /////////////HQPIK//////////////////////////
    const unsigned int hierarchy_num_hqpik_ = 3;
    const unsigned int variable_size_hqpik_ = 21;
	const unsigned int constraint_size1_hqpik_ = 21;	//[lb <=	x	<= 	ub] form constraints
	const unsigned int constraint_size2_hqpik_[3] = {12, 15, 17};	//[lb <=	Ax 	<=	ub] or [Ax = b]
	const unsigned int control_size_hqpik_[3] = {3, 14, 8};		//1: upperbody, 2: head + hand, 3: upperarm + shoulder AAC

    double w1_hqpik_[3];
    double w2_hqpik_[3];
    double w3_hqpik_[3];
    double w4_hqpik_[3];
    double w5_hqpik_[3];
    double w6_hqpik_[3];
    
    Eigen::MatrixXd H_hqpik_[3], A_hqpik_[3];
    Eigen::MatrixXd J_hqpik_[3], J_temp_;
    Eigen::VectorXd g_hqpik_[3], u_dot_hqpik_[3], qpres_hqpik_, ub_hqpik_[3],lb_hqpik_[3], ubA_hqpik_[3], lbA_hqpik_[3];
    Eigen::VectorXd q_dot_hqpik_[3];

    int last_solved_hierarchy_num_;
    const double equality_condition_eps_ = 1e-8;
    const double damped_puedoinverse_eps_ = 1e-5;
    ///////////////////////////////////////////////////
    
    /////////////HQPIK2//////////////////////////
    const int hierarchy_num_hqpik2_ = 3;
    const int variable_size_hqpik2_ = 21;
	const int constraint_size1_hqpik2_ = 21;	//[lb <=	x	<= 	ub] form constraints
	const int constraint_size2_hqpik2_[3] = {12, 16, 16};	//[lb <=	Ax 	<=	ub] or [Ax = b]
	const int control_size_hqpik2_[3] = {4, 12, 8};		//1: head ori(2)+pos(2), 2: hand(12), 3: upperarm(4)+shoulder(4)

    double w1_hqpik2_[3];   //task
    double w2_hqpik2_[3];   //kinetic energy
    double w3_hqpik2_[3];   //acceleration
    
    Eigen::MatrixXd H_hqpik2_[3], A_hqpik2_[3];
    Eigen::MatrixXd J_hqpik2_[3];
    Eigen::VectorXd g_hqpik2_[3], u_dot_hqpik2_[3], qpres_hqpik2_, ub_hqpik2_[3],lb_hqpik2_[3], ubA_hqpik2_[3], lbA_hqpik2_[3];
    Eigen::VectorXd q_dot_hqpik2_[3];

    bool hqpik2_verbose = false;
    bool hqpik2_print_constraints = false;

    bool sca_constraint_hqpik_ = false;
    bool sca_dynamic_version_ = false;
    const unsigned int num_sca_constraint_hqpik_ = 2;
    /////////////////////////////////////////////////////

    ////////////QP RETARGETING//////////////////////////////////
    const int variable_size_retargeting_ = 6;
	const int constraint_size1_retargeting_ = 6;	//[lb <=	x	<= 	ub] form constraints
	const int constraint_size2_retargeting_[3] = {6, 6, 9};	//[lb <=	Ax 	<=	ub] from constraints
   	const int control_size_retargeting_[3] = {3, 3, 3};		//1: left hand, 2: right hand, 3: relative hand

    Eigen::Vector3d robot_still_pose_lhand_, robot_t_pose_lhand_, robot_forward_pose_lhand_, robot_still_pose_rhand_, robot_t_pose_rhand_, robot_forward_pose_rhand_;
    Eigen::Vector3d lhand_mapping_vector_, rhand_mapping_vector_, lhand_mapping_vector_pre_, rhand_mapping_vector_pre_, lhand_mapping_vector_dot_, rhand_mapping_vector_dot_;
    Eigen::MatrixXd lhand_master_ref_stack_, lhand_robot_ref_stack_, rhand_master_ref_stack_, rhand_robot_ref_stack_, lhand_master_ref_stack_pinverse_, rhand_master_ref_stack_pinverse_;
    
    Eigen::MatrixXd H_retargeting_lhand_,  A_retargeting_lhand_, H_retargeting_rhand_, A_retargeting_rhand_;
    Eigen::VectorXd qpres_retargeting_[3], g_retargeting_lhand_, g_retargeting_rhand_, ub_retargeting_, lb_retargeting_, ubA_retargeting_[3], lbA_retargeting_[3];
    Eigen::MatrixXd E1_, E2_, E3_, H_retargeting_, A_retargeting_[3];
    Eigen::VectorXd g_retargeting_, u1_, u2_, u3_;

    Eigen::Vector3d h_d_lhand_, h_d_rhand_, h_pre_lhand_, h_pre_rhand_, r_pre_lhand_, r_pre_rhand_;
    double w1_retargeting_, w2_retargeting_, w3_retargeting_, human_shoulder_width_; 
    const double control_gain_retargeting_ = 100;
    const double human_vel_min_ = -2;
    const double human_vel_max_ = 2;
    const double w_dot_min_ = -30;
    const double w_dot_max_ = 30;

    ////////////////////////////////////////////////////////////
    Eigen::VectorXd stepping_input;
    Eigen::VectorXd stepping_input_;

    /////////////MPC-MJ//////////////////////////
    Eigen::Vector3d x_hat_;
    Eigen::Vector3d y_hat_;
    Eigen::Vector3d x_hat_p_;
    Eigen::Vector3d y_hat_p_;

    Eigen::Vector3d x_hat_thread_;
    Eigen::Vector3d y_hat_thread_;
    Eigen::Vector3d x_hat_p_thread_;
    Eigen::Vector3d y_hat_p_thread_;

    Eigen::Vector3d x_hat_thread2_;
    Eigen::Vector3d y_hat_thread2_;
    Eigen::Vector3d x_hat_p_thread2_;
    Eigen::Vector3d y_hat_p_thread2_;

    Eigen::VectorXd MPC_input_x_;
    Eigen::VectorXd MPC_input_y_;
    Eigen::Matrix3d A_mpc_;
    Eigen::Vector3d B_mpc_;
    Eigen::Vector3d C_mpc_transpose_;
    Eigen::MatrixXd P_ps_mpc_; 
    Eigen::MatrixXd P_pu_mpc_;
    Eigen::MatrixXd P_vs_mpc_; 
    Eigen::MatrixXd P_vu_mpc_;
    Eigen::MatrixXd P_zs_mpc_; 
    Eigen::MatrixXd P_zu_mpc_;
    Eigen::MatrixXd Q_prime_;
    Eigen::MatrixXd Q_mpc_;

    Eigen::VectorXd x_com_pos_recur_;
    Eigen::VectorXd x_com_vel_recur_;
    Eigen::VectorXd x_zmp_recur_;
    Eigen::VectorXd y_com_pos_recur_;
    Eigen::VectorXd y_com_vel_recur_;
    Eigen::VectorXd y_zmp_recur_;
    
    Eigen::VectorXd x_cp_recur_;
    Eigen::VectorXd y_cp_recur_;

    Eigen::MatrixXd F_cp_;
    Eigen::MatrixXd diff_matrix_;
    Eigen::MatrixXd F_zmp_;
    Eigen::MatrixXd H_cpmpc_;
    Eigen::MatrixXd H_cpStepping_mpc_;
    Eigen::MatrixXd weighting_cp_;
    Eigen::MatrixXd weighting_zmp_diff_;
    Eigen::VectorXd e1_cpmpc_;
    Eigen::VectorXd cpmpc_input_x_;
    Eigen::VectorXd cpmpc_deszmp_x_;
    Eigen::VectorXd cpmpc_input_y_;
    Eigen::VectorXd cpmpc_deszmp_y_;
    double force_temp_ = 0, theta_temp_ = 0;
    double ssp_flag_ = 0;
    Eigen::Vector2d dsp_scaler_dot_;
    Eigen::Vector2d dsp_scaler_;
    double dsp_scaler_x_dot_ = 0;
    double dsp_scaler_x_ = 0;
    double dsp_scaler_y_dot_ = 0;
    double dsp_scaler_y_ = 0;  
    double dsp_time_reducer_ = 0;
    double dsp_time_reducer_fixed_ = 0;
    double del_F_x_next_ = 0;
    double del_F_y_next_ = 0;
    double del_F_x_ = 0;
    double del_F_x_prev_ = 0;
    double del_F_x_LPF_ = 0;
    double del_F_y_ = 0;
    double del_F_x_thread_ = 0;
    double del_F_y_thread_ = 0;
    double del_F_x_next_thread_ = 0;
    double del_F_y_next_thread_ = 0;
    
    double cpmpc_des_zmp_x_thread_ = 0;
    double cpmpc_des_zmp_x_thread2_ = 0;

    ////////////AVATAR MODE PEDAL////////////////
    bool avatar_op_pedal_raw_ = false;
    bool avatar_op_pedal_ = false;
    bool avatar_op_pedal_pre_ = false;
    int pedal_click_in_1s_ = 0;
    double first_pedal_click_time_ = 0.0;

    void avatarModeStateMachine();
    void avatarUpperbodyModeUpdate(int mode_input);
    ////////////////////////////////////////////
    double cpmpc_des_zmp_y_thread_ = 0;    
    double cpmpc_des_zmp_y_thread2_ = 0;
    
    double cp_des_zmp_x_ = 0;
    double cp_des_zmp_x_prev_ = 0;
    double des_zmp_x_stepchange_ = 0;
    double des_zmp_x_prev_stepchange_ = 0;

    double cp_des_zmp_y_ = 0;
    double cp_des_zmp_y_prev_ = 0;
    double des_zmp_y_stepchange_ = 0;
    double des_zmp_y_prev_stepchange_ = 0;

    Eigen::Vector2d des_zmp_interpol_;
    // Thread 3
    Eigen::VectorXd U_x_mpc_;
    Eigen::VectorXd U_y_mpc_; 
    // Thread 2
    
    Eigen::Vector2d del_F_;
    Eigen::Vector3d x_hat_r_;
    Eigen::Vector3d x_hat_r_sc_;
    Eigen::Vector3d x_hat_r_p_sc_;
    Eigen::Vector3d y_hat_r_;
    Eigen::Vector3d y_hat_r_sc_;    
    Eigen::Vector3d y_hat_r_p_sc_;
    Eigen::Vector3d x_hat_r_p_;
    Eigen::Vector3d y_hat_r_p_;
    Eigen::Vector3d x_mpc_i_;
    Eigen::Vector3d y_mpc_i_; 
    Eigen::Vector3d x_diff_;
    Eigen::Vector3d y_diff_;
    Eigen::Vector2d cpmpc_diff_;
    Eigen::Vector2d cpStepping_diff_;

    int wieber_interpol_cnt_x_ = 0;
    int wieber_interpol_cnt_y_ = 0;
    int cpmpc_interpol_cnt_x_ = 0;
    int cpmpc_interpol_cnt_y_ = 0;
    bool mpc_x_update_ {false}, mpc_y_update_ {false} ;
    bool cpmpc_x_update_ {false}, cpmpc_y_update_ {false} ;
    double W1_mpc_ = 0, W2_mpc_ = 0, W3_mpc_ = 0;
    int alpha_step_mpc_ = 0;
    int alpha_step_mpc_thread_ = 0;

    Eigen::VectorXd alpha_mpc_;
    Eigen::VectorXd F_diff_mpc_x_;
    Eigen::VectorXd F_diff_mpc_y_;
    double alpha_lpf_ = 0;
    double temp_pos_y_ = 0;
    double F0_F1_mpc_x_ = 0, F1_F2_mpc_x_ = 0, F2_F3_mpc_x_ = 0, F0_F1_mpc_y_ = 0, F1_F2_mpc_y_ = 0, F2_F3_mpc_y_ = 0;
    Eigen::Vector2d foot_diff_current2next_;
    Eigen::Vector2d foot_diff_next2Nnext_;
    Eigen::Vector2d foot_diff_current2next_thread_;
    Eigen::Vector2d foot_diff_next2Nnext_thread_;
    Eigen::Vector2d foot_diff_current2next_mpc_;
    Eigen::Vector2d foot_diff_next2Nnext_mpc_;

    // Eigen::Vector2d foot_diff_currentTonext_;
    Eigen::Vector6d target_swing_foot;
    Eigen::Vector6d desired_swing_foot;
    Eigen::Vector6d desired_swing_foot_LPF_;
    Eigen::Vector2d del_F_LPF_;
    Eigen::Vector6d fixed_swing_foot;
    Eigen::Vector6d fixed_swing_foot_del_F_;
    Eigen::MatrixXd modified_del_zmp_; 
    Eigen::MatrixXd m_del_zmp_x;
    Eigen::MatrixXd m_del_zmp_y;
    double zmp_modif_time_margin_ = 0; 
    ////////////////////////////////////////////////////////////
    
    /////////////CAM-HQP//////////////////////////
    const int hierarchy_num_camhqp_ = 2;
    const int variable_size_camhqp_ = 8; // original number -> 6 (DG)
    const int constraint_size1_camhqp_ = 8; //[lb <=	x	<= 	ub] form constraints // original number -> 6 (DG)
    //const int constraint_size2_camhqp_[2] = {0, 3};	//[lb <=	Ax 	<=	ub] or [Ax = b]/ 0223 except Z axis control
    const int constraint_size2_camhqp_[2] = {0, 2};	//[lb <=	Ax 	<=	ub] or [Ax = b] 
    //const int control_size_camhqp_[2] = {3, 8}; //1: CAM control, 2: init pose // original number -> 6 (DG)
    const int control_size_camhqp_[2] = {2, 8}; //1: CAM control, 2: init pose // original number -> 6 (DG) / 0223 except Z axis control

    double w1_camhqp_[2];
    double w2_camhqp_[2];
    double w3_camhqp_[2];
    
    Eigen::MatrixXd H_camhqp_[2], A_camhqp_[2];
    Eigen::MatrixXd J_camhqp_[2];
    Eigen::VectorXd g_camhqp_[2], u_dot_camhqp_[2], qpres_camhqp_, ub_camhqp_[2],lb_camhqp_[2], ubA_camhqp_[2], lbA_camhqp_[2];
    Eigen::VectorXd q_dot_camhqp_[2];

    int control_joint_idx_camhqp_[8]; // original number -> 6 (DG)
    int last_solved_hierarchy_num_camhqp_;
    unsigned int torque_flag_x = 0, torque_flag_y = 0; 
    ///////////////////////////////////////////////////

    /////////////LEG-QPIK//////////////////////////
    const int hierarchy_num_leg_qpik_ = 1;
    const int variable_size_leg_qpik_ = 12;            // original number -> 6 (DG)
    const int constraint_size1_leg_qpik_ = 12;         //[lb <=	x	<= 	ub] form constraints // original number -> 6 (DG)
    const int constraint_size2_leg_qpik_ = 12; //[lb <=	Ax 	<=	ub] or [Ax = b]
    const int control_size_leg_qpik_ = 12;     //

    double w1_leg_qpik_;
    double w2_leg_qpik_;
    double w3_leg_qpik_;

    Eigen::MatrixXd H_leg_qpik_, A_leg_qpik_;
    Eigen::MatrixXd J_leg_qpik_;
    Eigen::VectorXd g_leg_qpik_, u_dot_leg_qpik_, qpres_leg_qpik_, ub_leg_qpik_, lb_leg_qpik_, ubA_leg_qpik_, lbA_leg_qpik_;
    Eigen::VectorXd q_dot_leg_qpik_;

    ///////////////////////////////////////////////////

    /////////////LEG-HQPIK//////////////////////////
    const int hierarchy_num_leg_hqpik_ = 2;
    const int variable_size_leg_hqpik_ = 12;            // original number -> 6 (DG)
    const int constraint_size1_leg_hqpik_ = 12;         //[lb <=	x	<= 	ub] form constraints // original number -> 6 (DG)
    const int constraint_size2_leg_hqpik_[2] = {12, 12}; //[lb <=	Ax 	<=	ub] or [Ax = b]
    const int control_size_leg_hqpik_[2] = {6, 6};     //1: leg position control, 2: leg orientation control

    double w1_leg_hqpik_[2];
    double w2_leg_hqpik_[2];
    double w3_leg_hqpik_[2];

    Eigen::MatrixXd H_leg_hqpik_[2], A_leg_hqpik_[2];
    Eigen::MatrixXd J_leg_hqpik_[2];
    Eigen::VectorXd g_leg_hqpik_[2], u_dot_leg_hqpik_[2], qpres_leg_hqpik_, ub_leg_hqpik_[2], lb_leg_hqpik_[2], ubA_leg_hqpik_[2], lbA_leg_hqpik_[2];
    Eigen::VectorXd q_dot_leg_hqpik_[2];

    int last_solved_hierarchy_num_leg_hqpik_;
    ///////////////////////////////////////////////////

    /////////////////////////MOB LEARNING LSTM///////////////////////////////////////////////////////
    // lstm c++
    struct LSTM
    {
        ~LSTM() { std::cout << "LSTM terminates" << std::endl; }
        std::atomic<bool> atb_lstm_input_update_{false};
        std::atomic<bool> atb_lstm_output_update_{false};

        int n_input;
        int n_sequence_length;
        int n_output;
        int n_hidden;

        int buffer_size;
        int nn_input_size;

        Eigen::VectorXd ring_buffer; //ring_buffer
        int buffer_head;
        int buffer_tail;

        int input_mode_idx;
        int output_mode_idx;

        Eigen::MatrixXd input_slow;
        Eigen::MatrixXd input_fast;
        Eigen::MatrixXd input_thread;
        Eigen::VectorXd input_mean;
        Eigen::VectorXd input_std;

        Eigen::VectorXd robot_input_data;

        Eigen::VectorXd real_output; //real out with physical dimension
        Eigen::VectorXd output_mean;
        Eigen::VectorXd output_std;

        Eigen::VectorXd W_ih;
        Eigen::VectorXd b_ih;
        Eigen::VectorXd W_hh;
        Eigen::VectorXd b_hh;
        Eigen::VectorXd W_linear;
        Eigen::VectorXd b_linear;

        Eigen::MatrixXd hidden;
        Eigen::MatrixXd cell;
        Eigen::VectorXd gates; // [input/update, forget, cell, output]

        Eigen::VectorXd input_gate;
        Eigen::VectorXd forget_gate;
        Eigen::VectorXd cell_gate;
        Eigen::VectorXd output_gate;
        Eigen::VectorXd output;
    };

    LSTM left_leg_mob_lstm_;
    LSTM right_leg_mob_lstm_;

    ifstream network_weights_file_[6];
    ifstream mean_std_file_[4];
    ifstream col_thr_file_;

    Eigen::VectorQd estimated_ext_torque_lstm_;
    Eigen::VectorQd maximum_collision_free_torque_;

    Eigen::VectorQd threshold_joint_torque_collision_;
    Eigen::VectorQd ext_torque_compensation_;

    int left_leg_collision_detected_link_;
    int left_leg_collision_cnt_[3];  //0: thigh, 1: lower leg, 2: foot 
    bool collision_detection_flag_ = false;
    int right_leg_collision_detected_link_;
    int right_leg_collision_cnt_[3];  //0: thigh, 1: lower leg, 2: foot 

    Eigen::Vector6d estimated_ext_force_lfoot_lstm_;
    Eigen::Vector6d estimated_ext_force_rfoot_lstm_;
    Eigen::Vector6d estimated_ext_force_lhand_lstm_;
    Eigen::Vector6d estimated_ext_force_rhand_lstm_;

    Eigen::VectorQd estimated_model_unct_torque_fast_;
    Eigen::VectorQd estimated_model_unct_torque_slow_;
    Eigen::VectorQd estimated_model_unct_torque_thread_;
    Eigen::VectorQd estimated_model_unct_torque_slow_lpf_;

    Eigen::VectorQd estimated_model_unct_torque_variance_fast_;
    Eigen::VectorQd estimated_model_unct_torque_variance_slow_;
    Eigen::VectorQd estimated_model_unct_torque_variance_thread_;


    void collectRobotInputData_acc_version();

    void loadLstmWeights(LSTM &lstm, std::string folder_path);
    void loadLstmMeanStd(LSTM &lstm, std::string folder_path);
    void initializeLegLSTM(LSTM &lstm);
    void calculateLstmInput(LSTM &lstm);
    void calculateLstmOutput(LSTM &lstm);

    void loadCollisionThreshold(std::string folder_path);

    // manual training data collection
    double last_stop_time_;
    bool stop_flag_;
    VectorQd stop_q_training_;


    /////////////////////////////////////////////////////////////////////////////////////////////////

    /////////////////////////PETER GRU///////////////////////////////////////////////////////
    // GRU c++
    struct GRU
    {
        ~GRU() { std::cout << "GRU terminates" << std::endl; }
        std::atomic<bool> atb_gru_input_update_{false};
        std::atomic<bool> atb_gru_output_update_{false};

        int n_input;
        int n_output;
        int n_hidden;

        int buffer_size;
        Eigen::VectorXd ring_buffer; //ring_buffer
        int buffer_head;
        int buffer_tail;

        int input_mode_idx;
        int output_mode_idx;

        Eigen::MatrixXd input_slow;
        Eigen::MatrixXd input_fast;
        Eigen::MatrixXd input_thread;
        Eigen::VectorXd input_mean;
        Eigen::VectorXd input_std;

        Eigen::VectorXd robot_input_data;

        Eigen::VectorXd output;
        Eigen::VectorXd real_output; //real out with physical dimension
        Eigen::VectorXd output_mean;
        Eigen::VectorXd output_std;

        Eigen::MatrixXd W_ih;   // [ W_ir | W_iz | W_in ] R^{3*n_hidden X n_input}
        Eigen::VectorXd b_ih;   // [ b_ir | b_iz | b_in ]
        Eigen::MatrixXd W_hh;   // [ W_hr | W_hz | W_hn ]
        Eigen::VectorXd b_hh;   // [ b_hr | b_hz | b_hn ]
        Eigen::MatrixXd W_linear;
        Eigen::VectorXd b_linear;

        Eigen::MatrixXd h_t;    // hidden
        Eigen::VectorXd r_t;    // reset
        Eigen::VectorXd z_t;    // update
        Eigen::VectorXd n_t;    // new gates

        ifstream network_weights_files[3];
        ifstream bias_files[3];
        ifstream mean_std_files[4];

        bool loadweightfile_verbose = false;
        bool loadmeanstdfile_verbose = true;
        bool gaussian_mode = true;
    };
    GRU left_leg_peter_gru_;
    GRU right_leg_peter_gru_;

    // ifstream network_weights_file_gru_[6];
    // ifstream mean_std_file_gru_[4];

    Eigen::VectorQd estimated_ext_torque_gru_;

    Eigen::Vector6d estimated_ext_force_lfoot_gru_;
    Eigen::Vector6d estimated_ext_force_rfoot_gru_;
    Eigen::Vector6d estimated_ext_force_lhand_gru_;
    Eigen::Vector6d estimated_ext_force_rhand_gru_;

    Eigen::VectorQd estimated_external_torque_gru_fast_;
    Eigen::VectorQd estimated_external_torque_gru_slow_;
    Eigen::VectorQd estimated_external_torque_gru_thread_;
    Eigen::VectorQd estimated_external_torque_gru_slow_lpf_;

    Eigen::VectorQd estimated_external_torque_variance_gru_fast_;
    Eigen::VectorQd estimated_external_torque_variance_gru_slow_;
    Eigen::VectorQd estimated_external_torque_variance_gru_thread_;

    void collectRobotInputData_peter_gru();

    void loadGruWeights(GRU &gru, std::string folder_path);
    void loadGruMeanStd(GRU &gru, std::string folder_path);

    void initializeLegGRU(GRU &gru, int n_input, int n_output, int n_hidden);
    void calculateGruInput(GRU &gru);
    void calculateGruOutput(GRU &gru);

    Eigen::VectorXd vecSigmoid(VectorXd input);
    Eigen::VectorXd vecTanh(VectorXd input);
    //////////////////////////////////////////////////////////////////////////////////////////

    //////////////Self Collision Avoidance Network////////////////
    struct MLP
    {
        ~MLP() { std::cout << "MLP terminate" << std::endl; }
        std::vector<Eigen::MatrixXd> weight;
        std::vector<Eigen::VectorXd> bias;
        std::vector<Eigen::VectorXd> hidden;
        std::vector<Eigen::MatrixXd> hidden_derivative;

        std::vector<std::string> w_path;
        std::vector<std::string> b_path;

        std::vector<ifstream> weight_files;
        std::vector<ifstream> bias_files;

        int n_input;
        int n_output;
        Eigen::VectorXd n_hidden;
        int n_layer;
        
        Eigen::VectorXd q_to_input_mapping_vector;

        Eigen::VectorXd input_slow;
        Eigen::VectorXd input_fast;
        Eigen::VectorXd input_thread;

        Eigen::VectorXd output_slow;
        Eigen::VectorXd output_fast;
        Eigen::VectorXd output_thread;

        Eigen::MatrixXd output_derivative_fast;

        Eigen::VectorXd hx_gradient_fast;
        Eigen::VectorXd hx_gradient_fast_lpf;
        Eigen::VectorXd hx_gradient_fast_pre;

        double hx;

        bool loadweightfile_verbose = false;
        bool loadbiasfile_verbose = false;

        int self_collision_stop_cnt_;
    }   larm_upperbody_sca_mlp_, rarm_upperbody_sca_mlp_, btw_arms_sca_mlp_;
    
    void setNeuralNetworks();
    void initializeScaMlp(MLP &mlp, int n_input, int n_output, Eigen::VectorXd n_hidden, Eigen::VectorXd q_to_input_mapping_vector);
    void loadScaNetwork(MLP &mlp, std::string folder_path);
    void calculateScaMlpInput(MLP &mlp);
    void calculateScaMlpOutput(MLP &mlp);
    void readWeightFile(MLP &mlp, int weight_num);
    void readBiasFile(MLP &mlp, int bias_num);

    std::atomic<bool> atb_mlp_input_update_{false};
    std::atomic<bool> atb_mlp_output_update_{false};

    Eigen::MatrixXd q_dot_buffer_slow_;  //20 stacks
    Eigen::MatrixXd q_dot_buffer_fast_;  //20 stacks
    Eigen::MatrixXd q_dot_buffer_thread_;  //20 stacks

    //////////////////////////////////////////////////////////////

private:
    Eigen::VectorQd ControlVal_;
        
    void initWalkingParameter();

    //Arm controller
    Eigen::VectorXd joint_limit_l_;
    Eigen::VectorXd joint_limit_h_;
    Eigen::VectorXd joint_vel_limit_l_;
    Eigen::VectorXd joint_vel_limit_h_;

    Eigen::Vector3d left_x_traj_pre_;
    Eigen::Vector3d right_x_traj_pre_;
    Eigen::Matrix3d left_rotm_pre_;
    Eigen::Matrix3d right_rotm_pre_;

    Eigen::VectorQVQd q_virtual_clik_;
    Eigen::Vector8d integral;

    bool first_loop_larm_;
    bool first_loop_rarm_;
    bool first_loop_upperbody_;
    bool first_loop_hqpik_;
    bool first_loop_hqpik2_;
    bool first_loop_qp_retargeting_;

    int printout_cnt_ = 0;
    bool first_loop_camhqp_;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////RL WH CustomCuntroller//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
public:

    const double hz_ =125.;
    const double pd_hz_ = 2000;
    double del_t = 1 / hz_;

    void loadNetwork();
    void processNoise();
    void processBias();
    void initBias();
    Eigen::Matrix<double, MODEL_DOF, 1> q_bias_;
    void processObservation();
    void feedforwardPolicy();
    MatrixXd conv_layer(const MatrixXd &input, const MatrixXd &weights, const MatrixXd &biases, int kernel_size, int stride);
    void feedforwardEncoder();
    void initVariable();

    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);

    static const int num_action = 12;
    static const int num_actuator_action = 12;
    static const int num_cur_state = 66;
    static const int num_cur_internal_state = 66;
    static const int num_state_skip = 2;
    static const int num_state_hist = 10;
    static const int num_state = num_cur_state * num_state_hist; // num_cur_internal_state*num_state_hist+num_action*(num_state_hist-1);
    static const int num_hidden1 = 512;
    static const int num_hidden2 = 512;
    static const int num_hidden3 = 128;

    Eigen::MatrixXd policy_net_w0_;
    Eigen::MatrixXd policy_net_b0_;
    Eigen::MatrixXd policy_net_w2_;
    Eigen::MatrixXd policy_net_b2_;
    Eigen::MatrixXd policy_net_w4_;
    Eigen::MatrixXd policy_net_b4_;
    Eigen::MatrixXd action_net_w_;
    Eigen::MatrixXd action_net_b_;
    Eigen::MatrixXd hidden_layer1_;
    Eigen::MatrixXd hidden_layer2_;
    Eigen::MatrixXd hidden_layer3_;
    Eigen::MatrixXd rl_action_;

    Eigen::MatrixXd value_net_w0_;
    Eigen::MatrixXd value_net_b0_;
    Eigen::MatrixXd value_net_w2_;
    Eigen::MatrixXd value_net_b2_;
    Eigen::MatrixXd value_net_w4_;
    Eigen::MatrixXd value_net_b4_;
    Eigen::MatrixXd value_net_w_;
    Eigen::MatrixXd value_net_b_;
    Eigen::MatrixXd value_hidden_layer1_;
    Eigen::MatrixXd value_hidden_layer2_;
    Eigen::MatrixXd value_hidden_layer3_;
    double value_;


    bool stop_by_value_thres_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;

    Eigen::MatrixXd state_;
    Eigen::MatrixXd state_cur_;
    Eigen::MatrixXd state_buffer_;
    Eigen::MatrixXd state_mean_;
    Eigen::MatrixXd state_var_;

    // encoder
    static const bool use_encoder_ = false;
    bool encoder_initialized = false;

    Eigen::MatrixXd state_history_;
    Eigen::MatrixXd encoder_input_;
    Eigen::MatrixXd encoder_output_;
    static const int history_len_ = 50;
    static const int history_skip_ = 5;
    static const int encoder_dim_ = 8;
    static const int encoder_conv1_kernel_ = 6;
    static const int encoder_conv1_stride_ = 3;
    static const int encoder_conv1_output_channel_ = 32;
    static const int encoder_conv1_input_channel_ = num_cur_state;
    static const int encoder_conv1_output_size_ = (history_len_-encoder_conv1_kernel_) / encoder_conv1_stride_ + 1;

    static const int encoder_conv2_kernel_ = 4;
    static const int encoder_conv2_stride_ = 2;
    static const int encoder_conv2_output_channel_ = 16;
    static const int encoder_conv2_input_channel_ = encoder_conv1_output_channel_;
    static const int encoder_conv2_output_size_ = (encoder_conv1_output_size_-encoder_conv2_kernel_) / encoder_conv2_stride_ + 1;

    static const int encoder_fc_input_ = 16*encoder_conv2_output_size_;

    Eigen::MatrixXd encoder_conv1_w_;
    Eigen::MatrixXd encoder_conv1_b_;
    Eigen::MatrixXd encoder_hidden_layer1_;
    Eigen::MatrixXd encoder_conv2_w_;
    Eigen::MatrixXd encoder_conv2_b_;
    Eigen::MatrixXd encoder_hidden_layer2_;
    Eigen::MatrixXd encoder_fc_w_;
    Eigen::MatrixXd encoder_fc_b_;
    void loadEncoderNetwork();

    Eigen::MatrixXd policy_input_;
    static const int policy_input_dim_ =  (use_encoder_) 
                                    ? num_state + encoder_dim_
                                    : num_state;
    std::ofstream writeFile;
    std::ofstream evalFile;
    std::ofstream actuator_data_file;
    bool actuator_net_log = true;

    float phase_ = 0.0;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_;
    Eigen::Vector12d q_leg_desired_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;
    double action_dt_accumulate_ = 0.0;

    Eigen::Vector3d euler_angle_;

    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    ros::Subscriber joy_sub_;

    // slider command
    Vector3_t base_lin_vel, base_ang_vel;

    double target_vel_x_ = 0.0;
    double pre_target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double pre_target_vel_y_ = 0.0;
    double target_heading_ = 0.0;
    double pre_target_vel_yaw_ = 0.0;
    double heading = 0.0;

    std::string base_path = "";
    std::string loadPathFromConfig(const std::string &config_file);
    void loadCommand(const std::string &command_file);


    // BIPED WALKING PARAMETER
    void walkingParameterSetting();
    const int number_of_foot_step = 2;
    Eigen::Vector3d phase_indicator_;
    Eigen::Vector3d step_length_x_;
    Eigen::Vector3d step_length_y_;
    Eigen::Vector3d step_yaw_;
    Eigen::Vector3d t_dsp_;
    Eigen::Vector3d t_ssp_;
    Eigen::Vector3d t_dsp_seconds;
    Eigen::Vector3d t_ssp_seconds;
    Eigen::Vector3d foot_height_;
    Eigen::Vector3d t_total_;
    int first_stance_foot_ = 1; // 1 means right foot stance, 0 means left foot stance
    const double com_height_ = 0.68;

    double t_last_;
    double t_start_;
    double t_temp_;  
    double t_double1_;
    double t_double2_;
    double zmp_offset = 0.02;



    // Policy input
    Eigen::VectorXd target_com_state_stance_frame_;
    Eigen::VectorXd target_swing_state_stance_frame_;


    Eigen::MatrixXd foot_step_;
    Eigen::MatrixXd foot_step_offset_;
    Eigen::MatrixXd foot_step_support_frame_;
    Eigen::MatrixXd foot_step_support_frame_offset_;

    Eigen::Isometry3d pelv_support_start_;
    // Eigen::Isometry3d pelv_support_init_;
    Eigen::Isometry3d pelv_support_init_yaw_;

    Eigen::Vector3d com_desired_;
    Eigen::Vector3d com_desired_dot_;

    Eigen::Vector3d com_support_init_yaw_;
    Eigen::Vector3d com_support_init_dot_yaw_;

    Eigen::Vector3d com_support_current_;
    Eigen::Vector3d com_support_current_dot_;
    Eigen::Vector3d com_support_current_dot_prev_;
    Eigen::Vector3d com_global_current_;
    Eigen::Vector3d com_global_current_dot_;
    Eigen::Vector3d com_global_current_dot_prev_;

    Eigen::Vector3d pelv_rpy_current_WH;
    Eigen::Vector3d rfoot_rpy_current_;
    Eigen::Vector3d lfoot_rpy_current_;
    Eigen::Isometry3d pelv_yaw_rot_current_from_global_WH;
    Eigen::Isometry3d rfoot_roll_rot_;
    Eigen::Isometry3d lfoot_roll_rot_;
    Eigen::Isometry3d rfoot_pitch_rot_;
    Eigen::Isometry3d lfoot_pitch_rot_;
    Eigen::Isometry3d rfoot_yaw_rot_;
    Eigen::Isometry3d lfoot_yaw_rot_;

    Eigen::Isometry3d pelv_global_current_;
    Eigen::Isometry3d lfoot_global_current_;
    Eigen::Isometry3d rfoot_global_current_;
    Eigen::Isometry3d pelv_global_init_;
    Eigen::Isometry3d lfoot_global_init_;
    Eigen::Isometry3d rfoot_global_init_;

    Eigen::Isometry3d pelv_trajectory_support_; //local frame
    Eigen::Isometry3d pelv_trajectory_support_fast_; //local frame
    Eigen::Isometry3d pelv_trajectory_support_slow_; //local frame

    Eigen::Isometry3d rfoot_trajectory_support_;  //local frame
    Eigen::Isometry3d lfoot_trajectory_support_;
    Eigen::Isometry3d lfoot_trajectory_support_fast_;
    Eigen::Isometry3d lfoot_trajectory_support_slow_;

    Eigen::Vector3d rfoot_trajectory_euler_support_;
    Eigen::Vector3d lfoot_trajectory_euler_support_;

    Eigen::Isometry3d pelv_trajectory_float_; //pelvis frame

    Eigen::Vector3d com_trajectory_float_;

    Eigen::Isometry3d lfoot_trajectory_float_;
    Eigen::Isometry3d lfoot_trajectory_float_fast_;
    Eigen::Isometry3d lfoot_trajectory_float_slow_;

    Eigen::Isometry3d rfoot_trajectory_float_;
    Eigen::Isometry3d rfoot_trajectory_float_fast_;
    Eigen::Isometry3d rfoot_trajectory_float_slow_;

    Eigen::Vector3d pelv_support_euler_init_yaw_;
    Eigen::Vector3d lfoot_support_euler_init_yaw_;
    Eigen::Vector3d rfoot_support_euler_init_yaw_;
    double wn = sqrt(GRAVITY / com_height_);

    double walking_end_flag = 0;

    Eigen::Isometry3d swingfoot_global_current_; 
    Eigen::Isometry3d supportfoot_global_current_; // Only yaw is considered

    Eigen::Isometry3d pelv_support_current_;
    Eigen::Isometry3d lfoot_support_current_;
    Eigen::Isometry3d rfoot_support_current_;

    Eigen::Isometry3d lfoot_support_init_yaw_;
    Eigen::Isometry3d rfoot_support_init_yaw_;

    Eigen::Isometry3d target_com_state_float_frame_, target_lfoot_state_float_frame_, target_rfoot_state_float_frame_;


    Eigen::MatrixXd ref_zmp_;
    Eigen::MatrixXd ref_zmp_container;
    Eigen::MatrixXd ref_zmp_thread3;
    Eigen::VectorXd ref_com_yaw_;
    Eigen::VectorXd ref_com_yawvel_;
    // PREVIEW CONTROL
    Eigen::Vector3d x_preview_;
    Eigen::Vector3d y_preview_;

    Eigen::Vector3d xs_preview_;
    Eigen::Vector3d ys_preview_;
    Eigen::Vector3d xd_preview_;
    Eigen::Vector3d yd_preview_; 

    Eigen::MatrixXd Gi_preview_;
    Eigen::MatrixXd Gx_preview_;
    Eigen::VectorXd Gd_preview_;
    Eigen::MatrixXd A_preview_;
    Eigen::VectorXd B_preview_;
    Eigen::MatrixXd C_preview_;
    double UX_preview_, UY_preview_, EX_preview_, EY_preview_;


    Eigen::Vector6d l_ft_;
    Eigen::Vector6d r_ft_;
    Eigen::Vector6d l_ft_LPF;
    Eigen::Vector6d r_ft_LPF;
    Eigen::Vector2d zmp_measured_mj_;
    Eigen::Vector2d zmp_measured_LPF_;

    double P_angle_i = 0;
    double P_angle = 0;
    double P_angle_input_dot = 0;
    double P_angle_input = 0;
    double R_angle = 0;
    double R_angle_input_dot = 0;
    double R_angle_input = 0;
    double aa = 0; 
    double Y_angle_input = 0;

    // BOOLEAN OPERATOR
    bool walking_enable_;
    bool is_lfoot_support = false;
    bool is_rfoot_support = false;
    bool is_dsp1 = false;
    bool is_ssp  = false;
    bool is_dsp2 = false;
    bool is_preview_ctrl_init = true;

    // Logging
    Eigen::Isometry3d supportfoot_global_init_; // Used just to compute init_yaw_
    Eigen::Isometry3d supportfoot_global_init_yaw_;
    Eigen::VectorXd swing_state_stance_frame_;
    Eigen::VectorXd com_state_stance_frame_;

    // USER COMMAND
    double Lcommand_step_length_x_ = 0.;
    double Lcommand_step_length_y_ = 0.22;
    double Lcommand_step_yaw_ = 0.;
    double Lcommand_t_dsp_ = 0.1;
    double Lcommand_t_ssp_ = .8;
    double Lcommand_foot_height_ = 0.12;

    double Rcommand_step_length_x_ = 0.;
    double Rcommand_step_length_y_ = 0.22;
    double Rcommand_step_yaw_ = 0.;
    double Rcommand_t_dsp_ = 0.1;
    double Rcommand_t_ssp_ = .8;
    double Rcommand_foot_height_ = 0.12;

    bool ideal_preview = false;

    bool eval_mode = false; // Genearate a fixed set of commands. For comparison between methods.
    int current_step_number = 0;
    int planned_step_number = 12;
    Eigen::VectorXd step_length_x_planned;
    Eigen::VectorXd step_length_y_planned;
    Eigen::VectorXd step_yaw_planned;
    Eigen::VectorXd t_dsp_planned;
    Eigen::VectorXd t_ssp_planned;
    Eigen::VectorXd foot_height_planned;


    int iter_x_l=0;
    int iter_x_r=0;
    int iter_y_l_ls =0;
    int iter_y_l_rs =0;
    int iter_y_r_ls =0;
    int iter_y_r_rs =0;
    int iter_y_r =0;
    int iter_yaw_l_ls=0;
    int iter_yaw_l_rs=0;
    int iter_yaw_r_ls=0;
    int iter_yaw_r_rs=0;
    int iter_end_x=0;
    int iter_end_y_l_ls =0;
    int iter_end_y_l_rs =0;
    int iter_end_y_r_ls =0;
    int iter_end_y_r_rs =0;
    int iter_end_y_r =0;
    int iter_end_yaw_l=0;
    int iter_end_yaw_r=0;
    double joy_x;
    double joy_y;
    double joy_height = 0.12;
    double joy_length =0.0;
    double joy_length_y =0.22;
    double joy_length_temp =0.2;
    double joy_length_y_temp =0.35;
    double joy_length_command =0.;
    double joy_length_y_r_command =0.;
    double joy_length_y_l_command =0.;
    double joy_length_previous=0.0;
    double joy_length_r =0.;
    double joy_length_l =0.;
    double joy_length_y_l =0.22;
    double joy_length_y_r =0.22;
    double joy_length_y_r_temp =0.22;
    double joy_length_y_l_temp =0.22;
    double joy_length_y_l_previous =0.22;
    double joy_length_y_r_previous =0.22;
    double joy_yaw_r =0.;
    double joy_yaw_l =0.;
    double joy_yaw_l_previous =0.;
    double joy_yaw_r_previous =0.;
    double joy_yaw_r_command =0.;
    double joy_yaw_l_command =0.;
    double swing_previous_l =0.;
    double swing_previous_r =0.;
    double joy_temp;


    double joy_left_foot_pos;
    double joy_right_foot_pos;
    double joy_walking_tick;
    double previous_l =0.;
    double previous_r =0.;
    double current_l =0.;
    double current_r =0.;
    double previous_lswing =0.;
    double previous_rswing =0.;
    double current_lswing =0.;
    double current_rswing =0.;


    std::vector<int> last_buttons_0;
    std::vector<int> last_buttons_1;
    std::vector<int> last_buttons_2;
    std::vector<int> last_buttons_3;
    std::vector<double> joy_walking_tick_temp;
    std::vector<double> joy_length_previous_vec= std::vector<double>(10,0.0);;
    std::vector<double> swing_previous_r_vec;
    std::vector<double> swing_previous_l_vec;
    std::vector<double> joy_yaw_l_previous_vec = std::vector<double>(10,0.0);
    std::vector<double> joy_yaw_r_previous_vec= std::vector<double>(10,0.0);
    std::vector<double> joy_length_y_r_previous_vec= std::vector<double>(10,0.22);;
    std::vector<double> joy_length_y_l_previous_vec= std::vector<double>(10,0.22);
    std::vector<double> last_buttons_7;

    double x_error = 0;
    double y_error = 0;
    double yaw_error = 0;

    double norm;

    // PREVIEW CONTROL
    void updateInitialState();
    void updateFootstepCommand();
    void getRobotState();
    void walkingStateMachine();
    void calculateFootStepTotal();
    void supportToFloatPattern();
    void floatToSupportFootstep();
    void updateNextStepTime();
    void resetPreviewState();
    void windupPreview();
    void computeIkControl(const Eigen::Isometry3d &float_trunk_transform, const Eigen::Isometry3d &float_lleg_transform, const Eigen::Isometry3d &float_rleg_transform, Eigen::Vector12d &q_des);

    void getZmpTrajectory();
    void addZmpOffset();
    void zmpGenerator(const unsigned int norm_size);
    void onestepZmp(unsigned int current_step_number, Eigen::VectorXd &temp_px, Eigen::VectorXd &temp_py, Eigen::VectorXd& temp_yaw, Eigen::VectorXd &temp_yawvel);

    void getComTrajectory(); 
    void previewcontroller(double dt, int NL, int tick, 
                        Eigen::Vector3d &x_k, Eigen::Vector3d &y_k, double &UX, double &UY,
                        const Eigen::MatrixXd &Gi, const Eigen::VectorXd &Gd, const Eigen::MatrixXd &Gx, 
                        const Eigen::MatrixXd &A,  const Eigen::VectorXd &B,  const Eigen::MatrixXd &C);
    void preview_Parameter(double dt, int NL, Eigen::MatrixXd& Gi, Eigen::VectorXd& Gd, Eigen::MatrixXd& Gx, Eigen::MatrixXd& A, Eigen::VectorXd& B, Eigen::MatrixXd& C);
    void getComTrajectory_mpc();
    void getFootTrajectory(); 
    void getTargetState(); 
    
private:    
    unsigned int initial_flag = 0;
    unsigned int walking_tick = 0;

};
