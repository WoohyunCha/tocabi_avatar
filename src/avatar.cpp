#include "avatar.h"
#include <fstream>
using namespace TOCABI;

AvatarController::AvatarController(RobotData &rd) 
: rd_(rd)
{
    nh_avatar_.setCallbackQueue(&queue_avatar_);

    upperbodymode_sub = nh_avatar_.subscribe("/tocabi/avatar/upperbodymodecommand", 100, &AvatarController::UpperbodyModeCallback, this);

    arm_pd_gain_sub = nh_avatar_.subscribe("/tocabi/dg/armpdgain", 100, &AvatarController::ArmJointGainCallback, this);
    waist_pd_gain_sub = nh_avatar_.subscribe("/tocabi/dg/waistpdgain", 100, &AvatarController::WaistJointGainCallback, this);

    //subscribers
    tracker_status_sub = nh_avatar_.subscribe("/TRACKERSTATUS", 100, &AvatarController::TrackerStatusCallback, this);
    tracker_pose_sub = nh_avatar_.subscribe("/tracker_pose", 1, &AvatarController::TrackerPoseCallback, this, ros::TransportHints().tcpNoDelay(true));
    master_pose_sub = nh_avatar_.subscribe("/tocabi/haptic_poses", 1, &AvatarController::MasterPoseCallback, this, ros::TransportHints().tcpNoDelay(true));

    vive_tracker_pose_calibration_sub = nh_avatar_.subscribe("/tocabi/avatar/pose_calibration_flag", 100, &AvatarController::PoseCalibrationCallback, this);
    robot_hand_pos_mapping_scale_sub = nh_avatar_.subscribe("/tocabi/avatar/hand_pos_mapping_sclae", 100, &AvatarController::HandPosMappingScaleCallback, this);

    //publishers
    calibration_state_pub = nh_avatar_.advertise<std_msgs::String>("/tocabi_status", 5);
    calibration_state_gui_log_pub = nh_avatar_.advertise<std_msgs::String>("/tocabi/guilog", 100);
    upperbodymode_pub = nh_avatar_.advertise<std_msgs::Int8>("/tocabi/avatar/upperbodymodecurrent", 5); 
    avatar_warning_pub = nh_avatar_.advertise<std_msgs::Int8>("/tocabi/avatar/warningmsg", 5); 

    haptic_force_pub = nh_avatar_.advertise<std_msgs::Float32MultiArray>("/tocabi/hand_ftsensors", 5);

    bool urdfmode = false;
    std::string urdf_path, desc_package_path;
    ros::param::get("/tocabi_controller/urdf_path", desc_package_path);

    RigidBodyDynamics::Addons::URDFReadFromFile(desc_package_path.c_str(), &model_d_, true, false);
    RigidBodyDynamics::Addons::URDFReadFromFile(desc_package_path.c_str(), &model_C_, true, false);
    RigidBodyDynamics::Addons::URDFReadFromFile(desc_package_path.c_str(), &model_global_, true, false);
    RigidBodyDynamics::Addons::URDFReadFromFile(desc_package_path.c_str(), &model_local_, true, false);
    RigidBodyDynamics::Addons::URDFReadFromFile(desc_package_path.c_str(), &model_MJ_, true, false);

    for (int i = 0; i < FILE_CNT; i++)
    {
        file[i].open(FILE_NAMES[i]);
    }

    setGains();
    setNeuralNetworks();
    first_loop_larm_ = true;
    first_loop_rarm_ = true;
    first_loop_upperbody_ = true;
    first_loop_hqpik_ = true;
    first_loop_hqpik2_ = true;
    first_loop_qp_retargeting_ = true;
    first_loop_camhqp_ = true;

    // RL-based Walking Initialization
    initVariable();
    loadNetwork();
    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("joy", 10, &AvatarController::joyCallback, this);
}

void AvatarController::setGains()
{
    /// For Real Robot
    kp_stiff_joint_.setZero();
    kv_stiff_joint_.setZero();
    kp_stiff_joint_.diagonal() << 2000, 5000, 4000, 3700, 5000, 5000,
                                  2000, 5000, 4000, 3700, 5000, 5000,
                                  6000, 10000, 10000,
                                  200, 400, 200, 200, 125, 125, 25, 25,
                                  50, 50,
                                  200, 400, 200, 200, 125, 125, 25, 25;
    kv_stiff_joint_.diagonal() <<   15, 50, 20, 25, 30, 30,
                                    15, 50, 20, 25, 30, 30,
                                    200, 100, 100,
                                    7, 5, 2.5, 2.5, 2.5, 2, 2, 2,
                                    2, 2,
                                    7, 5, 2.5, 2.5, 2.5, 2, 2, 2;

    for (int i = 0; i < MODEL_DOF; i++)
    {
        kp_joint_(i) = kp_stiff_joint_(i,i);
        kv_joint_(i) = kv_stiff_joint_(i,i);
    }

    // arm controller
    joint_limit_l_.resize(33);
    joint_limit_h_.resize(33);
    joint_vel_limit_l_.resize(33);
    joint_vel_limit_h_.resize(33);

    // LEG
    for (int i = 0; i < 12; i++)
    {
        joint_limit_l_(i) = -180 * DEG2RAD;
        joint_limit_h_(i) = 180 * DEG2RAD;
    }

    // WAIST
    joint_limit_l_(12) = -30 * DEG2RAD;
    joint_limit_h_(12) = 30 * DEG2RAD;
    joint_limit_l_(13) = -15 * DEG2RAD;
    joint_limit_h_(13) = 30 * DEG2RAD;
    joint_limit_l_(14) = -15 * DEG2RAD;
    joint_limit_h_(14) = 15 * DEG2RAD;
    // LEFT ARM
    joint_limit_l_(15) = -30 * DEG2RAD;
    joint_limit_h_(15) = 30 * DEG2RAD;
    joint_limit_l_(16) = -160 * DEG2RAD;
    joint_limit_h_(16) = 90 * DEG2RAD;
    joint_limit_l_(17) = -95 * DEG2RAD;
    joint_limit_h_(17) = 95 * DEG2RAD;
    joint_limit_l_(18) = -180 * DEG2RAD;
    joint_limit_h_(18) = 180 * DEG2RAD;
    joint_limit_l_(19) = -150 * DEG2RAD;
    joint_limit_h_(19) = -15 * DEG2RAD;
    joint_limit_l_(20) = -180 * DEG2RAD;
    joint_limit_h_(20) = 180 * DEG2RAD;
    joint_limit_l_(21) = -70 * DEG2RAD;
    joint_limit_h_(21) = 70 * DEG2RAD;
    joint_limit_l_(22) = -60 * DEG2RAD;
    joint_limit_h_(22) = 60 * DEG2RAD;
    // HEAD
    joint_limit_l_(23) = -80 * DEG2RAD;
    joint_limit_h_(23) = 80 * DEG2RAD;
    joint_limit_l_(24) = -40 * DEG2RAD;
    joint_limit_h_(24) = 18 * DEG2RAD;
    // RIGHT ARM
    joint_limit_l_(25) = -30 * DEG2RAD;
    joint_limit_h_(25) = 30 * DEG2RAD;
    joint_limit_l_(26) = -90 * DEG2RAD;
    joint_limit_h_(26) = 160 * DEG2RAD;
    joint_limit_l_(27) = -95 * DEG2RAD;
    joint_limit_h_(27) = 95 * DEG2RAD;
    joint_limit_l_(28) = -180 * DEG2RAD;
    joint_limit_h_(28) = 180 * DEG2RAD;
    joint_limit_l_(29) = 15 * DEG2RAD;
    joint_limit_h_(29) = 150 * DEG2RAD;
    joint_limit_l_(30) = -180 * DEG2RAD;
    joint_limit_h_(30) = 180 * DEG2RAD;
    joint_limit_l_(31) = -70 * DEG2RAD;
    joint_limit_h_(31) = 70 * DEG2RAD;
    joint_limit_l_(32) = -60 * DEG2RAD;
    joint_limit_h_(32) = 60 * DEG2RAD;

    // LEG
    for (int i = 0; i < 12; i++)
    {
        joint_vel_limit_l_(i) = -2 * M_PI;
        joint_vel_limit_h_(i) = 2 * M_PI;
    }

    // UPPERBODY
    for (int i = 12; i < 33; i++)
    {
        joint_vel_limit_l_(i) = -M_PI * 2;
        joint_vel_limit_h_(i) = M_PI * 2;
    }

    // 1st arm joint vel limit
    joint_vel_limit_l_(15) = -M_PI / 3;
    joint_vel_limit_h_(15) = M_PI / 3;

    joint_vel_limit_l_(25) = -M_PI / 3;
    joint_vel_limit_h_(25) = M_PI / 3;

    // Head joint vel limit
    joint_vel_limit_l_(23) = -2 * M_PI;
    joint_vel_limit_h_(23) = 2 * M_PI;
    joint_vel_limit_l_(24) = -2 * M_PI;
    joint_vel_limit_h_(24) = 2 * M_PI;

    // forearm joint vel limit
    joint_vel_limit_l_(20) = -2 * M_PI;
    joint_vel_limit_h_(20) = 2 * M_PI;
    joint_vel_limit_l_(30) = -2 * M_PI;
    joint_vel_limit_h_(30) = 2 * M_PI;
}
void AvatarController::setNeuralNetworks()
{
    ///// Between Left Arm and Upperbody & Head Collision Detection Network /////
    Eigen::VectorXd n_hidden, q_to_input_mapping_vector;
    n_hidden.resize(6);
    q_to_input_mapping_vector.resize(13);
    n_hidden << 120, 100, 80, 60, 40, 20;
    q_to_input_mapping_vector << 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24;
    initializeScaMlp(larm_upperbody_sca_mlp_, 13, 2, n_hidden, q_to_input_mapping_vector);
    // loadScaNetwork(larm_upperbody_sca_mlp_, "/home/dyros/catkin_ws/src/tocabi_avatar/sca_mlp/larm_upperbody/");
    loadScaNetwork(larm_upperbody_sca_mlp_, "/home/cha/catkin_ws/src/tocabi_avatar/sca_mlp/larm_upperbody/");
    //////////////////////////////////////////////////////////////////////////////

    ///// Between Right Arm and Upperbody & Head Collision Detection Network /////
    n_hidden << 120, 100, 80, 60, 40, 20;
    q_to_input_mapping_vector << 12, 13, 14, 25, 26, 27, 28, 29, 30, 31, 32, 23, 24;
    initializeScaMlp(rarm_upperbody_sca_mlp_, 13, 2, n_hidden, q_to_input_mapping_vector);
    loadScaNetwork(rarm_upperbody_sca_mlp_, "/home/cha/catkin_ws/src/tocabi_avatar/sca_mlp/rarm_upperbody/");
    //////////////////////////////////////////////////////////////////////////////

    ///// Between Arms Collision Detection Network /////
    // q_to_input_mapping_vector.resize(16);
    // n_hidden << 120, 100, 80, 60, 40, 20;
    // q_to_input_mapping_vector << 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32;
    // initializeScaMlp(btw_arms_sca_mlp_, 16, 2, n_hidden, q_to_input_mapping_vector);
    // loadScaNetwork(btw_arms_sca_mlp_, "/home/dyros/catkin_ws/src/tocabi_avatar/sca_mlp/btw_arms/");
    //////////////////////////////////////////////////////////////////////////////
}
Eigen::VectorQd AvatarController::getControl()
{
    return rd_.torque_desired;
}

void AvatarController::computeSlow()
{
    queue_avatar_.callAvailable(ros::WallDuration());
    copyRobotData(rd_);

    if (rd_.tc_.mode == 12)
    {
        if (initial_flag == 0)
        {
            desired_q_slow_ = rd_.q_;
            desired_q_fast_ = rd_.q_;
            desired_q_dot_.setZero();
            desired_q_dot_fast_.setZero();

            initWalkingParameter();
            loadCollisionThreshold("/home/dyros/catkin_ws/src/tocabi_avatar/config/");
            cout << "mode = 12 : Pedal Init" << endl;
            cout << "chair_mode_: " << chair_mode_ << endl;
            initial_flag = 1;
        }

        for (int i = 0; i < MODEL_DOF; i++)
        {
            rd_.torque_desired(i) = kp_joint_(i) * (desired_q_slow_(i) - rd_.q_(i)) - kv_joint_(i) * rd_.q_dot_(i); // + 1.0 * Gravity_MJ_(i);
        }

    }
    else if (rd_.tc_.mode == 13) //2KHZ STRICT
    {
        if (initial_flag == 1)
        {

            for (int i=0; i<LINK_NUMBER + 1; i++){
                link_avatar_[i] = rd_.link_[i];
            }
            // for (int i = 0; i < 12; i++){
            //     Initial_current_q_(i) = rd_.q_[i];
            // }

            //Initialize settings for Task Control! 

            start_time_ = rd_cc_.control_time_us_;

            q_noise_pre_ = q_noise_ = q_init_ = rd_cc_.q_virtual_.segment(6,MODEL_DOF);

            q_leg_desired_ = rd_cc_.q_.segment(0,12);

            time_cur_ = start_time_ / 1e6;

            time_pre_ = time_cur_ - 0.005;

            // time_inference_pre_ = rd_cc_.control_time_us_ - (1/249.9)*1e6;

            time_inference_pre_ = rd_cc_.control_time_us_ - (1/(hz_))*1e6;



            rd_.tc_init = false;

            std::cout<<"cc mode 7"<<std::endl;

            torque_init_ = rd_cc_.torque_desired;

            target_com_state_stance_frame_.setZero(13);

            target_swing_state_stance_frame_.setZero(13);

            updateFootstepCommand();

            getRobotState();

            walkingStateMachine();

            getComTrajectory(); 

            getFootTrajectory();



            getTargetState();

            processNoise();

            // Woohyun

            processBias();

            processObservation();

            for (int i = 0; i < num_state_skip*num_state_hist; i++) 

            {

                state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1) = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();

                // state_buffer_.block(num_cur_state*i, 0, num_cur_state, 1).setZero();

            }
            /////////////////////////////////////////////////////
            initial_flag = 2;
        }



        processNoise();

        // Woohyun

        processBias();


        // processObservation and feedforwardPolicy mean time: 15 us, max 53 us
        if ((rd_cc_.control_time_us_ - time_inference_pre_)/1.0e6 >= 1/hz_ )
        {
            // auto start_time = std::chrono::high_resolution_clock::now();



            // Call the functions you want to measure

            updateFootstepCommand();

            getRobotState();

            walkingStateMachine();

            getComTrajectory(); 

            getFootTrajectory();



            getTargetState();



            processObservation();

            feedforwardPolicy();



            updateNextStepTime();



            // End time measurement

            // auto end_time = std::chrono::high_resolution_clock::now();



            // // Calculate the duration in microseconds

            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();



            // // Output the time taken

            // std::cout << "processObservation and feedforwardPolicy took " << duration << " us" << std::endl;



            

            // action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5/250.0, 0.0, 5/250.0);

            action_dt_accumulate_ += DyrosMath::minmax_cut(rl_action_(num_action-1)*5/hz_, 0.0, 5/hz_);

            if (value_ < 50.0)

            {

                if (stop_by_value_thres_ == false)

                {



                    stop_by_value_thres_ = true;

                    stop_start_time_ = rd_cc_.control_time_us_;

                    q_stop_ = q_noise_;

                    std::cout << "Stop by Value Function : " << walking_tick << ", Value : " << value_ << std::endl;

                }

            }         

            time_inference_pre_ = rd_cc_.control_time_us_;
        }
        
        torque_lower_.setZero();
        for (int i = 0; i < num_actuator_action; i++)
        {
            torque_lower_(i) = DyrosMath::minmax_cut(rl_action_(i)*torque_bound_(i), -torque_bound_(i), torque_bound_(i));
        }

        if (atb_desired_q_update_ == false)
        {
            atb_desired_q_update_ = true;
            desired_q_fast_ = desired_q_slow_;
            desired_q_dot_fast_ = desired_q_dot_slow_;
            atb_desired_q_update_ = false;
        }

        torque_upper_.setZero();
        for (int i = 12; i < MODEL_DOF; i++)
        {
            torque_upper_(i) = (kp_joint_(i) * (desired_q_fast_(i) - rd_.q_(i)) + kv_joint_(i) * (desired_q_dot_fast_(i) - rd_.q_dot_(i)));// + 1.0 * Gravity_MJ_fast_(i));
            rd_.q_desired(i) = desired_q_fast_(i);  // for logging
            rd_.q_dot_desired(i) = desired_q_dot_fast_(i);  // for logging
        }
        ///////////////////////////////FINAL TORQUE COMMAND/////////////////////////////
        rd_.torque_desired = torque_lower_ + torque_upper_;
        // if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        // {
        //     for (int i = 0; i <MODEL_DOF; i++)
        //     {
        //         torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_, start_time_, start_time_ + 0.1e6, torque_init_(i), torque_lower_(i), 0.0, 0.0);
        //     }
        //     rd_.torque_desired = torque_spline_ + torque_upper_;
        // }
        // else
        // {
        //     rd_.torque_desired = torque_lower_ + torque_upper_;
        // }
        if (stop_by_value_thres_)
        {
            rd_.torque_desired = kp_stiff_joint_ * (q_stop_ - q_noise_) - kv_stiff_joint_ * q_vel_noise_;
        }
        ////////////////////////////////////////////////////////////////////////////////
    }
    else if (rd_.tc_.mode == 14)
    {

    }
}

void AvatarController::computeFast()
{
    if (rd_.tc_.mode == 13)
    {
        if (rd_.tc_init == true)
        {
            initWalkingParameter();
            rd_.tc_init = false;
        }

        // data process//
        getRobotData(); // 47~64us
        getProcessedRobotData(); // <<1us
        // motion planing and control//

        // Self-Collision-Avoidance Network Inferences
        calculateScaMlpOutput(larm_upperbody_sca_mlp_);
        calculateScaMlpOutput(rarm_upperbody_sca_mlp_);
        // avatar mode pedal
        avatarModeStateMachine();

        //motion planing and control//
        motionGenerator(); // 140~240us(HQPIK)

        for (int i = 12; i < MODEL_DOF; i++)
        {
            desired_q_(i) = motion_q_(i);
            desired_q_dot_(i) = motion_q_dot_(i); 
        }

        //STEP4: send desired q to the fast thread
        if (atb_desired_q_update_ == false)
        {
            atb_desired_q_update_ = true;
            desired_q_slow_ = desired_q_;
            desired_q_dot_slow_ = desired_q_dot_;
            atb_desired_q_update_ = false;
        }
        savePreData();

    }
    else if (rd_.tc_.mode == 14)
    {
    }
}

void AvatarController::computeThread3()
{
}

void AvatarController::initWalkingParameter()
{
    walking_mode_on_ = true;
    upper_body_mode_ = 3;
    upper_body_mode_raw_ = 3;

    upperbody_mode_recieved_ = true;

    pre_time_ = rd_.control_time_ - 0.001;
    pre_desired_q_ = rd_.q_;
    last_desired_q_ = rd_.q_;
    pre_desired_q_dot_.setZero();

    init_q_ = rd_.q_;
    zero_q_ = init_q_;
    desired_q_ = init_q_;
    desired_q_dot_.setZero();
    desired_q_ddot_.setZero();


    motion_q_pre_ = init_q_;
    motion_q_dot_pre_.setZero();


    lhand_control_point_offset_.setZero();
    rhand_control_point_offset_.setZero();
    // lhand_control_point_offset_(2) = -0.13;
    // rhand_control_point_offset_(2) = -0.13;
    lhand_control_point_offset_(2) = -0.13; // without FT
    rhand_control_point_offset_(2) = -0.15; // with FT

    robot_shoulder_width_ = 0.6;

    robot_upperarm_max_l_ = 0.3376 * 1.0;
    robot_lowerarm_max_l_ = 0.31967530867;
    // robot_arm_max_l_ = 0.98*sqrt(robot_upperarm_max_l_*robot_upperarm_max_l_ + robot_lowerarm_max_l_*robot_lowerarm_max_l_ + 2*robot_upperarm_max_l_*robot_lowerarm_max_l_*cos( -joint_limit_h_(19)) );
    robot_arm_max_l_ = (robot_upperarm_max_l_ + robot_lowerarm_max_l_) * 0.95 + lhand_control_point_offset_.norm();

    hmd_check_pose_calibration_[0] = false;
    hmd_check_pose_calibration_[1] = false;
    hmd_check_pose_calibration_[2] = false;
    hmd_check_pose_calibration_[3] = false;
    hmd_check_pose_calibration_[4] = false;

    still_pose_cali_flag_ = false;
    t_pose_cali_flag_ = false;
    forward_pose_cali_flag_ = false;
    read_cali_log_flag_ = false;

    hmd_larm_max_l_ = 0.45;
    hmd_rarm_max_l_ = 0.45;
    hmd_shoulder_width_ = 0.5;

    hmd_pelv_pose_.setIdentity();
    hmd_lshoulder_pose_.setIdentity();
    hmd_lhand_pose_.setIdentity();
    hmd_rshoulder_pose_.setIdentity();
    hmd_rupperarm_pose_.setIdentity();
    hmd_rhand_pose_.setIdentity();
    hmd_chest_pose_.setIdentity();

    hmd_pelv_pose_raw_.setIdentity();
    hmd_lshoulder_pose_raw_.setIdentity();
    hmd_lhand_pose_raw_.setIdentity();
    hmd_rshoulder_pose_raw_.setIdentity();
    hmd_rupperarm_pose_raw_.setIdentity();
    hmd_rhand_pose_raw_.setIdentity();
    hmd_chest_pose_raw_.setIdentity();

    hmd_head_pose_raw_last_.setIdentity();
    hmd_pelv_pose_raw_last_.setIdentity();
    hmd_lshoulder_pose_raw_last_.setIdentity();
    hmd_lupperarm_pose_raw_last_.setIdentity();
    hmd_lhand_pose_raw_last_.setIdentity();
    hmd_rshoulder_pose_raw_last_.setIdentity();
    hmd_rupperarm_pose_raw_last_.setIdentity();
    hmd_rhand_pose_raw_last_.setIdentity();
    hmd_chest_pose_raw_last_.setIdentity();
    

    hmd_head_pose_pre_.setIdentity();
    hmd_lshoulder_pose_pre_.setIdentity();
    hmd_lupperarm_pose_pre_.setIdentity();
    hmd_lhand_pose_pre_.setIdentity();
    hmd_rshoulder_pose_pre_.setIdentity();
    hmd_rupperarm_pose_pre_.setIdentity();
    hmd_rhand_pose_pre_.setIdentity();
    hmd_chest_pose_pre_.setIdentity();
    hmd_pelv_pose_pre_.setIdentity();

    hmd_pelv_pose_init_.setIdentity();
    tracker_status_changed_time_ = current_time_;
    hmd_tracker_status_ = false;
    hmd_tracker_status_raw_ = false;
    hmd_tracker_status_pre_ = false;

    hmd_head_abrupt_motion_count_ = 0;
    hmd_lupperarm_abrupt_motion_count_ = 0;
    hmd_lhand_abrupt_motion_count_ = 0;
    hmd_rupperarm_abrupt_motion_count_ = 0;
    hmd_rhand_abrupt_motion_count_ = 0;
    hmd_chest_abrupt_motion_count_ = 0;
    hmd_pelv_abrupt_motion_count_ = 0;

    last_solved_hierarchy_num_ = hierarchy_num_hqpik_ - 1;
}

void AvatarController::getRobotData()
{
    current_time_ = (double)rd_.control_time_us_ / 1000000.0;

    if (current_time_ != pre_time_)
    {
        dt_ = current_time_ - pre_time_;
    }

    if(dt_ < 0)
    {
        // cout<< cred << "WARNING: 'dt' is negative in thread2: "<< dt_ << creset << endl;
        current_time_ = pre_time_;
    }
    else if(dt_ > 0.002)
    {
        // cout<< cred <<"WARNING: 'dt' is too large in thread2: "<< dt_<< creset << endl;
    }

    dt_ = DyrosMath::minmax_cut(dt_, 0.0005, 0.002);

    current_q_ = __q_virtual.segment(6,MODEL_DOF);
    current_q_dot_ = __q_dot_virtual.segment(6,MODEL_DOF);
    current_q_ddot_ = __q_ddot_virtual.segment(6, MODEL_DOF);
    pelv_pos_current_ = link_avatar_[Pelvis].xpos;
    pelv_vel_current_.segment(0, 3) = link_avatar_[Pelvis].v;
    pelv_vel_current_.segment(3, 3) = link_avatar_[Pelvis].w;

    pelv_rot_current_ = link_avatar_[Pelvis].rotm;
    pelv_rpy_current_ = DyrosMath::rot2Euler(pelv_rot_current_); // ZYX multiply
    // pelv_rpy_current_ = (pelv_rot_current_).eulerAngles(2, 1, 0);
    // pelv_yaw_rot_current_from_global_ = DyrosMath::rotateWithZ(pelv_rpy_current_(2));
    pelv_yaw_rot_current_from_global_ = pelv_rot_current_;
    pelv_rot_current_yaw_aline_ = pelv_yaw_rot_current_from_global_.transpose() * pelv_rot_current_;
    // pelv_pos_current_ = pelv_yaw_rot_current_from_global_.transpose() * pelv_pos_current_;

    pelv_transform_current_from_global_.translation().setZero();
    pelv_transform_current_from_global_.linear() = pelv_rot_current_yaw_aline_;

    pelv_angvel_current_ = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Pelvis].w;

    com_mass_ = link_avatar_[COM_id].mass;

    /////////////////////////Feet Transformation and Velocity/////////////////////
    lfoot_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Foot].xpos - pelv_pos_current_);
    lfoot_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Foot].rotm;
    rfoot_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Foot].xpos - pelv_pos_current_);
    rfoot_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Foot].rotm;

    lfoot_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Foot].v;
    lfoot_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Foot].w;
    rfoot_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Foot].v;
    rfoot_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Foot].w;
    ///////////////////////////////////////////////////////////////////////////////

    ////////////////////////Knee Trnasformation and Velocity///////////////////////
    lknee_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Foot - 2].xpos - pelv_pos_current_);
    lknee_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Foot - 2].rotm;
    rknee_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Foot - 2].xpos - pelv_pos_current_);
    rknee_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Foot - 2].rotm;
    ///////////////////////////////////////////////////////////////////////////////

    ////////////////////////Hand Trnasformation and Velocity///////////////////////
    lhand_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Hand].xpos - pelv_pos_current_);
    lhand_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand].rotm;
    rhand_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Hand].xpos - pelv_pos_current_);
    rhand_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand].rotm;

    lhand_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand].v;
    lhand_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand].w;
    rhand_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand].v;
    rhand_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand].w;
    ///////////////////////////////////////////////////////////////////////////////

    ////////////////////////Elbow Trnasformation and Velocity///////////////////////
    lelbow_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Hand - 3].xpos - pelv_pos_current_);
    lelbow_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 3].rotm;
    relbow_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Hand - 3].xpos - pelv_pos_current_);
    relbow_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 3].rotm;

    lelbow_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 3].v;
    lelbow_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 3].w;
    relbow_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 3].v;
    relbow_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 3].w;
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////Upper Arm Trnasformation and Velocity////////////////////
    lupperarm_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Hand - 4].xpos - pelv_pos_current_);
    lupperarm_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 4].rotm;
    rupperarm_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Hand - 4].xpos - pelv_pos_current_);
    rupperarm_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 4].rotm;

    lupperarm_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 4].v;
    lupperarm_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 4].w;
    rupperarm_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 4].v;
    rupperarm_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 4].w;
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////Shoulder Trnasformation and Velocity////////////////////
    lshoulder_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Hand - 5].xpos - pelv_pos_current_);
    lshoulder_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 5].rotm;
    rshoulder_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Hand - 5].xpos - pelv_pos_current_);
    rshoulder_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 5].rotm;

    lshoulder_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 5].v;
    lshoulder_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 5].w;
    rshoulder_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 5].v;
    rshoulder_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 5].w;
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////Acromion Trnasformation and Velocity////////////////////
    lacromion_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Hand - 6].xpos - pelv_pos_current_);
    lacromion_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 6].rotm;
    racromion_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Hand - 6].xpos - pelv_pos_current_);
    racromion_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 6].rotm;

    lacromion_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 6].v;
    lacromion_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 6].w;
    racromion_vel_current_from_global_.segment(0, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 6].v;
    racromion_vel_current_from_global_.segment(3, 3) = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 6].w;
    ////////////////////////////////////////////////////////////////////////////////

    ///////////////////////Armbase Trasformation and ///////////////////////////////
    larmbase_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Left_Hand - 7].xpos - pelv_pos_current_);
    larmbase_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Left_Hand - 7].rotm;
    rarmbase_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Right_Hand - 7].xpos - pelv_pos_current_);
    rarmbase_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Right_Hand - 7].rotm;
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////Head & Upperbody Trnasformation ////////////////////////
    head_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Head].xpos - pelv_pos_current_);
    head_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Head].rotm;

    upperbody_transform_current_from_global_.translation() = pelv_yaw_rot_current_from_global_.transpose() * (link_avatar_[Upper_Body].xpos - pelv_pos_current_);
    upperbody_transform_current_from_global_.linear() = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[Upper_Body].rotm;
    ////////////////////////////////////////////////////////////////////////////////

    ///////////////////////Rotation Euler Angles////////////////////////////////////
    lhand_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(lhand_transform_current_from_global_.linear());
    rhand_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(rhand_transform_current_from_global_.linear());
    lelbow_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(lelbow_transform_current_from_global_.linear());
    relbow_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(relbow_transform_current_from_global_.linear());
    lupperarm_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(lshoulder_transform_current_from_global_.linear());
    rupperarm_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(rupperarm_transform_current_from_global_.linear());
    lshoulder_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(lupperarm_transform_current_from_global_.linear());
    rshoulder_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(rshoulder_transform_current_from_global_.linear());
    lacromion_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(lacromion_transform_current_from_global_.linear());
    racromion_rpy_current_from_global_ = DyrosMath::rot2Euler_tf(racromion_transform_current_from_global_.linear());
    ////////////////////////////////////////////////////////////////////////////////

    ///////////////////////Variables Updated from desired joint position////////////////////////////
    pre_desired_q_qvqd_;
    Quaterniond q(pelv_rot_current_yaw_aline_); // conversion error

    pre_desired_q_qvqd_.setZero();
    // pre_desired_q_qvqd_(3) = q.x(); 	//x y z
    // pre_desired_q_qvqd_(4) = q.y(); 	//x y z
    // pre_desired_q_qvqd_(5) = q.z(); 	//x y z
    // pre_desired_q_qvqd_(39) = q.w();						//w
    pre_desired_q_qvqd_(39) = 1;
    pre_desired_q_qvqd_.segment(6, MODEL_DOF) = pre_desired_q_;

    pre_desired_q_dot_vqd_.setZero();
    pre_desired_q_dot_vqd_.segment(0, 6) = pelv_vel_current_;
    pre_desired_q_dot_vqd_.segment(6, MODEL_DOF) = pre_desired_q_dot_;

    pre_desired_q_ddot_vqd_.setZero();

    VectorXd q_ddot_virtual, q_dot_virtual, q_virtual;
    q_virtual = pre_desired_q_qvqd_;
    q_dot_virtual = pre_desired_q_dot_vqd_;
    q_ddot_virtual = pre_desired_q_ddot_vqd_;
    RigidBodyDynamics::UpdateKinematicsCustom(model_d_, &q_virtual, &q_dot_virtual, &q_ddot_virtual);

    lfoot_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Foot].id, Eigen::Vector3d::Zero(), false);
    lfoot_transform_pre_desired_from_.linear() = (RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Foot].id, false)).transpose();

    lhand_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand].id, lhand_control_point_offset_, false);
    lhand_transform_pre_desired_from_.linear() = (RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand].id, false)).transpose();

    lelbow_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 3].id, Eigen::Vector3d::Zero(), false);
    lelbow_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 3].id, false).transpose();

    lupperarm_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 4].id, Eigen::Vector3d::Zero(), false);
    lupperarm_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 4].id, false).transpose();

    lshoulder_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 5].id, Eigen::Vector3d::Zero(), false);
    lshoulder_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 5].id, false).transpose();

    lacromion_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 6].id, Eigen::Vector3d::Zero(), false);
    lacromion_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 6].id, false).transpose();

    larmbase_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 7].id, Eigen::Vector3d::Zero(), false);
    larmbase_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 7].id, false).transpose();

    rhand_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand].id, rhand_control_point_offset_, false);
    rhand_transform_pre_desired_from_.linear() = (RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand].id, false)).transpose();

    relbow_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 3].id, Eigen::Vector3d::Zero(), false);
    relbow_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 3].id, false).transpose();

    rupperarm_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 4].id, Eigen::Vector3d::Zero(), false);
    rupperarm_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 4].id, false).transpose();

    rshoulder_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 5].id, Eigen::Vector3d::Zero(), false);
    rshoulder_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 5].id, false).transpose();

    racromion_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 6].id, Eigen::Vector3d::Zero(), false);
    racromion_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 6].id, false).transpose();

    rarmbase_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 7].id, Eigen::Vector3d::Zero(), false);
    rarmbase_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 7].id, false).transpose();

    head_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Head].id, Eigen::Vector3d::Zero(), false);
    head_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Head].id, false).transpose();

    upperbody_transform_pre_desired_from_.translation() = RigidBodyDynamics::CalcBodyToBaseCoordinates(model_d_, pre_desired_q_qvqd_, link_avatar_[Upper_Body].id, Eigen::Vector3d::Zero(), false);
    upperbody_transform_pre_desired_from_.linear() = RigidBodyDynamics::CalcBodyWorldOrientation(model_d_, pre_desired_q_qvqd_, link_avatar_[Upper_Body].id, false).transpose();

    RigidBodyDynamics::Math::Vector3d com_pos_temp, com_vel_temp, com_accel_temp, com_ang_momentum_temp, com_ang_moment_temp;

    RigidBodyDynamics::Utils::CalcCenterOfMass(model_d_, q_virtual, q_dot_virtual, &q_ddot_virtual, com_mass_, com_pos_temp, &com_vel_temp, &com_accel_temp, &com_ang_momentum_temp, &com_ang_moment_temp, false);
    ///////////////////////////////////////////////////////////////////////////////////////////

    Matrix6d R_R;
    R_R.setZero();
    R_R.block(0, 0, 3, 3) = pelv_yaw_rot_current_from_global_.transpose();
    R_R.block(3, 3, 3, 3) = pelv_yaw_rot_current_from_global_.transpose();
    // R_R.setIdentity();

    jac_com_ = R_R * link_avatar_[COM_id].jac.cast<double>();
    jac_com_pos_ = pelv_yaw_rot_current_from_global_.transpose() * link_avatar_[COM_id].jac_com.cast<double>().topRows(3);
}

void AvatarController::getProcessedRobotData()
{

    if (walking_mode_on_) // command on
    {
        start_time_ = current_time_;
        program_start_time_ = current_time_;

        init_q_ = current_q_;
        last_desired_q_ = current_q_;

        walking_mode_on_ = false;

        pelv_transform_init_from_support_ = pelv_transform_current_from_support_;
        pelv_transform_start_from_support_ = pelv_transform_current_from_support_;
        lfoot_transform_start_from_support_ = lfoot_transform_current_from_support_;
        rfoot_transform_start_from_support_ = rfoot_transform_current_from_support_;

        lfoot_transform_desired_ = lfoot_transform_current_from_support_;
        rfoot_transform_desired_ = rfoot_transform_current_from_support_;
        pelv_transform_desired_ = pelv_transform_current_from_support_;

        lfoot_transform_desired_last_ = lfoot_transform_current_from_support_;
        rfoot_transform_desired_last_ = rfoot_transform_current_from_support_;
        pelv_transform_desired_last_ = pelv_transform_current_from_support_;

        pelv_transform_start_from_global_.translation() = pelv_pos_current_;
        pelv_transform_start_from_global_.linear() = pelv_rot_current_yaw_aline_;
        lfoot_transform_start_from_global_ = lfoot_transform_current_from_global_;
        rfoot_transform_start_from_global_ = rfoot_transform_current_from_global_;

        lfoot_transform_init_from_global_ = lfoot_transform_current_from_global_;
        rfoot_transform_init_from_global_ = rfoot_transform_current_from_global_;

        lhand_transform_init_from_global_ = lhand_transform_current_from_global_;
        rhand_transform_init_from_global_ = rhand_transform_current_from_global_;

        lelbow_transform_init_from_global_ = lelbow_transform_current_from_global_;
        relbow_transform_init_from_global_ = relbow_transform_current_from_global_;

        lupperarm_transform_init_from_global_ = lupperarm_transform_current_from_global_;
        rupperarm_transform_init_from_global_ = rupperarm_transform_current_from_global_;

        lshoulder_transform_init_from_global_ = lshoulder_transform_current_from_global_;
        rshoulder_transform_init_from_global_ = rshoulder_transform_current_from_global_;

        lacromion_transform_init_from_global_ = lacromion_transform_current_from_global_;
        racromion_transform_init_from_global_ = racromion_transform_current_from_global_;

        larmbase_transform_init_from_global_ = larmbase_transform_current_from_global_;
        rarmbase_transform_init_from_global_ = rarmbase_transform_current_from_global_;

        head_transform_init_from_global_ = head_transform_current_from_global_;
        upperbody_transform_init_from_global_ = upperbody_transform_current_from_global_;

        master_lhand_pose_raw_pre_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_raw_pre_ = rhand_transform_pre_desired_from_;
        master_lelbow_pose_raw_pre_ = lupperarm_transform_pre_desired_from_;
        master_relbow_pose_raw_pre_ = rupperarm_transform_pre_desired_from_;
        master_lshoulder_pose_raw_pre_ = lacromion_transform_pre_desired_from_;
        master_rshoulder_pose_raw_pre_ = racromion_transform_pre_desired_from_;
        master_head_pose_raw_pre_ = head_transform_pre_desired_from_;
        master_upperbody_pose_raw_pre_ = upperbody_transform_pre_desired_from_;

        master_lhand_pose_raw_ppre_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_raw_ppre_ = rhand_transform_pre_desired_from_;
        master_head_pose_raw_ppre_ = head_transform_pre_desired_from_;
        master_lelbow_pose_raw_ppre_ = lupperarm_transform_pre_desired_from_;
        master_relbow_pose_raw_ppre_ = rupperarm_transform_pre_desired_from_;
        master_lshoulder_pose_raw_ppre_ = lacromion_transform_pre_desired_from_;
        master_rshoulder_pose_raw_ppre_ = racromion_transform_pre_desired_from_;
        master_upperbody_pose_raw_ppre_ = upperbody_transform_pre_desired_from_;

        master_lhand_pose_pre_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_pre_ = rhand_transform_pre_desired_from_;
        master_lelbow_pose_pre_ = lupperarm_transform_pre_desired_from_;
        master_relbow_pose_pre_ = rupperarm_transform_pre_desired_from_;
        master_lshoulder_pose_pre_ = lacromion_transform_pre_desired_from_;
        master_rshoulder_pose_pre_ = racromion_transform_pre_desired_from_;
        master_head_pose_pre_ = head_transform_pre_desired_from_;
        master_upperbody_pose_pre_ = upperbody_transform_pre_desired_from_;

        master_lhand_pose_ppre_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_ppre_ = rhand_transform_pre_desired_from_;
        master_head_pose_ppre_ = head_transform_pre_desired_from_;
        master_lelbow_pose_ppre_ = lupperarm_transform_pre_desired_from_;
        master_relbow_pose_ppre_ = rupperarm_transform_pre_desired_from_;
        master_lshoulder_pose_ppre_ = lacromion_transform_pre_desired_from_;
        master_rshoulder_pose_ppre_ = racromion_transform_pre_desired_from_;
        master_upperbody_pose_ppre_ = upperbody_transform_pre_desired_from_;

        master_relative_lhand_pos_pre_ = lhand_transform_current_from_global_.translation() - rhand_transform_current_from_global_.translation();
        master_relative_rhand_pos_pre_ = rhand_transform_current_from_global_.translation() - lhand_transform_current_from_global_.translation();

        master_lhand_vel_.setZero();
        master_rhand_vel_.setZero();
        master_lelbow_vel_.setZero();
        master_relbow_vel_.setZero();
        master_lshoulder_vel_.setZero();
        master_rshoulder_vel_.setZero();

        master_lhand_rqy_.setZero();
        master_rhand_rqy_.setZero();
        master_lelbow_rqy_.setZero();
        master_relbow_rqy_.setZero();
        master_lshoulder_rqy_.setZero();
        master_rshoulder_rqy_.setZero();
        master_head_rqy_.setZero();

        lhand_vel_error_.setZero();
        rhand_vel_error_.setZero();
        lelbow_vel_error_.setZero();
        relbow_vel_error_.setZero();
        lacromion_vel_error_.setZero();
        racromion_vel_error_.setZero();

        pelv_pos_init_ = pelv_pos_current_;
        pelv_vel_init_ = pelv_vel_current_;
        pelv_rot_init_ = pelv_rot_current_;
        pelv_rpy_init_ = pelv_rpy_current_;
        pelv_rot_init_yaw_aline_ = pelv_rot_current_yaw_aline_;
        pelv_transform_init_from_global_ = pelv_transform_current_from_global_;

        lfoot_transform_init_from_global_ = lfoot_transform_current_from_global_;
        rfoot_transform_init_from_global_ = rfoot_transform_current_from_global_;
        // lfoot_transform_init_from_global_ = lfoot_transform_pre_desired_from_;
        // rfoot_transform_init_from_global_ = rfoot_transform_pre_desired_from_;


        // init_q_ = current_q_;
        last_desired_q_ = desired_q_;


        lfoot_transform_init_from_support_ = lfoot_transform_current_from_support_;
        rfoot_transform_init_from_support_ = rfoot_transform_current_from_support_;
        pelv_transform_init_from_support_ = pelv_transform_current_from_support_;
        pelv_rpy_init_from_support_ = DyrosMath::rot2Euler(pelv_transform_init_from_support_.linear());
    }
}

void AvatarController::avatarModeStateMachine()
{
    //////CHECK SELF COLLISION///////////
    if(larm_upperbody_sca_mlp_.hx < 0.0)
    {
        larm_upperbody_sca_mlp_.self_collision_stop_cnt_ += 1;
    }
    else
    {
        larm_upperbody_sca_mlp_.self_collision_stop_cnt_ == 0;
    }

    if(rarm_upperbody_sca_mlp_.hx < 0.0)
    {
        rarm_upperbody_sca_mlp_.self_collision_stop_cnt_ += 1;
    }
    else
    {
        rarm_upperbody_sca_mlp_.self_collision_stop_cnt_ == 0;
    }

    if(upper_body_mode_ != 3)
    {
        if(larm_upperbody_sca_mlp_.self_collision_stop_cnt_ > 100 && current_time_ > upperbody_command_time_ + 3.0)
        {
            avatarUpperbodyModeUpdate(3);

            larm_upperbody_sca_mlp_.self_collision_stop_cnt_ = 0;
            cout<< cred << "WARNING: Self Collision is Detected btw Left Arm - Body" << creset << endl;

            std_msgs::Int8 warning_msg_1;
            warning_msg_1.data = 1;
            avatar_warning_pub.publish(warning_msg_1);

            std_msgs::String msg;
            std::stringstream larm_selfcol;
            larm_selfcol << "Self Collision (Left Arm)";
            msg.data = larm_selfcol.str();
            calibration_state_gui_log_pub.publish(msg);
        }
        if(rarm_upperbody_sca_mlp_.self_collision_stop_cnt_ > 100 && current_time_ > upperbody_command_time_ + 3.0)
        {
            avatarUpperbodyModeUpdate(3);

            rarm_upperbody_sca_mlp_.self_collision_stop_cnt_ = 0;
            cout<< cred << "WARNING: Self Collision is Detected btw Right Arm - Body" << creset << endl;

            std_msgs::Int8 warning_msg_2;
            warning_msg_2.data = 2;
            avatar_warning_pub.publish(warning_msg_2);

            std_msgs::String msg;
            std::stringstream rarm_selfcol;
            rarm_selfcol << "Self Collision (Right Arm)";
            msg.data = rarm_selfcol.str();
            calibration_state_gui_log_pub.publish(msg);
        }
    }

    //test
    // if( int(rd_.control_time_*2000)%1000 == 0)
    // {
    //     cout<<"larm hx: "<<larm_upperbody_sca_mlp_.hx << endl;
    //     cout<<"rarm hx: "<<rarm_upperbody_sca_mlp_.hx << endl;
    // }
    //////////////////////////////////////////////////////

    /// @brief masterarm haptic feedback publihser
    double time_smooting = 3.0;
    
    if( upper_body_mode_ < 6)
    {
        if( current_time_ > upperbody_command_time_+time_smooting)
        {
            lh_ft_feedback_.setZero();
            rh_ft_feedback_.setZero();
        }
        else
        {
            double linear_spline;
            linear_spline = DyrosMath::minmax_cut( 
                1-(current_time_ - upperbody_command_time_)/time_smooting, 0.0, 1.0 );
            
            lh_ft_feedback_ = linear_spline*lh_ft_feedback_;
            rh_ft_feedback_ = linear_spline*rh_ft_feedback_;
        }
    }
    else
    {
        if( current_time_ <= upperbody_command_time_+time_smooting)
        {
            double linear_spline;
            linear_spline = DyrosMath::minmax_cut( 
                (current_time_ - upperbody_command_time_)/time_smooting, 0.0, 1.0 );
            
            lh_ft_feedback_ = linear_spline*lh_ft_feedback_;
            rh_ft_feedback_ = linear_spline*rh_ft_feedback_;
        }
    }
    
    upper_body_mode_ = upper_body_mode_raw_;

    /// @brief upper body mode publisher for GUI
    std_msgs::Int8 msg;
    msg.data = upper_body_mode_;
    upperbodymode_pub.publish(msg);
}
void AvatarController::avatarUpperbodyModeUpdate(int mode_input)
{
    upper_body_mode_raw_ = mode_input;
    upperbody_mode_recieved_ = true;
    // upperbody_command_time_ = current_time_;
    // upperbody_mode_q_init_ = motion_q_pre_;
}
void AvatarController::motionGenerator()
{
    motion_q_dot_.setZero();
    motion_q_.setZero();
    pd_control_mask_.setZero();

    ///////////////////////LEG/////////////////////////
    //////LEFT LEG///////0 0 0.02 0.15 -0.17 0
    motion_q_(0) = 0;
    motion_q_(1) = 0;
    motion_q_(2) = 0.02;
    motion_q_(3) = 0.6;
    motion_q_(4) = -0.12;
    motion_q_(5) = 0;
    pd_control_mask_(0) = 1;
    pd_control_mask_(1) = 0;
    pd_control_mask_(2) = 0;
    pd_control_mask_(3) = 1;
    pd_control_mask_(4) = 1;
    pd_control_mask_(5) = 1;
    //////////////////////
    /////RIFHT LEG////////0 0 0.02 0.15 -0.17 0
    motion_q_(6) = 0;
    motion_q_(7) = 0;
    motion_q_(8) = 0.02;
    motion_q_(9) = 0.6;
    motion_q_(10) = -0.12;
    motion_q_(11) = 0;
    pd_control_mask_(6) = 1;
    pd_control_mask_(7) = 0;
    pd_control_mask_(8) = 0;
    pd_control_mask_(9) = 1;
    pd_control_mask_(10) = 1;
    pd_control_mask_(11) = 1;
    //////////////////////

    poseCalibration();

    if (upper_body_mode_ == 1) // init pose
    {
        if (upperbody_mode_recieved_ == true)
        {
            cout << "Upperbody Mode is Changed to #1" << endl;
            upperbody_mode_recieved_ = false;
            upperbody_command_time_ = current_time_;
            upperbody_mode_q_init_ = motion_q_pre_;
        }

        ///////////////////////WAIST/////////////////////////
        motion_q_(12) = 0;
        motion_q_(13) = 0; // pitch
        motion_q_(14) = 0; // roll
        pd_control_mask_(12) = 1;
        pd_control_mask_(13) = 1;
        pd_control_mask_(14) = 1;
        /////////////////////////////////////////////////////

        ///////////////////////HEAD/////////////////////////
        motion_q_(23) = 0; // yaw
        motion_q_(24) = 0; // pitch
        pd_control_mask_(23) = 1;
        pd_control_mask_(24) = 1;
        /////////////////////////////////////////////////////

        ///////////////////////ARM/////////////////////////
        //////LEFT ARM///////0.3 0.3 1.5 -1.27 -1 0 -1 0
        motion_q_(15) = 0.3;
        motion_q_(16) = 0.3;
        motion_q_(17) = 1.5;
        motion_q_(18) = -1.27;
        motion_q_(19) = -1.0;
        motion_q_(20) = 0.0;
        motion_q_(21) = -1.0;
        motion_q_(22) = 0.0;

        pd_control_mask_(15) = 1;
        pd_control_mask_(16) = 1;
        pd_control_mask_(17) = 1;
        pd_control_mask_(18) = 1;
        pd_control_mask_(19) = 1;
        pd_control_mask_(20) = 1;
        pd_control_mask_(21) = 1;
        pd_control_mask_(22) = 1;
        //////////////////////
        /////RIFHT ARM////////-0.3 -0.3 -1.5 1.27 1 0 1 0
        motion_q_(25) = -0.3;
        motion_q_(26) = -0.3;
        motion_q_(27) = -1.5;
        motion_q_(28) = 1.27;
        motion_q_(29) = 1.0;
        motion_q_(30) = 0.0;
        motion_q_(31) = 1.0;
        motion_q_(32) = 0.0;

        pd_control_mask_(25) = 1;
        pd_control_mask_(26) = 1;
        pd_control_mask_(27) = 1;
        pd_control_mask_(28) = 1;
        pd_control_mask_(29) = 1;
        pd_control_mask_(30) = 1;
        pd_control_mask_(31) = 1;
        pd_control_mask_(32) = 1;
        /////////////////////////////////////////////////////

        for (int i = 12; i < MODEL_DOF; i++)
        {
            motion_q_(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + 4, upperbody_mode_q_init_(i), 0, 0, motion_q_(i), 0, 0)(0);
        }
    }
    else if (upper_body_mode_ == 2) // Zero pose
    {
        if (upperbody_mode_recieved_ == true)
        {
            cout << "Upperbody Mode is Changed to #2" << endl;
            upperbody_mode_recieved_ = false;
            upperbody_command_time_ = current_time_;
            upperbody_mode_q_init_ = motion_q_pre_;
        }
        ///////////////////////WAIST/////////////////////////
        motion_q_(12) = 0; // pitch
        motion_q_(13) = 0; // pitch
        motion_q_(14) = 0; // roll
        pd_control_mask_(12) = 1;
        pd_control_mask_(13) = 1;
        pd_control_mask_(14) = 1;
        /////////////////////////////////////////////////////

        ///////////////////////HEAD/////////////////////////
        motion_q_(23) = 0; // yaw
        motion_q_(24) = 0; // pitch
        pd_control_mask_(23) = 1;
        pd_control_mask_(24) = 1;
        /////////////////////////////////////////////////////

        ///////////////////////ARM/////////////////////////
        //////LEFT ARM///////0.3 0.3 1.5 -1.27 -1 0 -1 0
        motion_q_(15) = 0.3;
        motion_q_(16) = 0.12;
        motion_q_(17) = 1.43;
        motion_q_(18) = -0.85;
        motion_q_(19) = -0.45; // elbow
        motion_q_(20) = 1;
        motion_q_(21) = 0.0;
        motion_q_(22) = 0.0;
        pd_control_mask_(15) = 1;
        pd_control_mask_(16) = 1;
        pd_control_mask_(17) = 1;
        pd_control_mask_(18) = 1;
        pd_control_mask_(19) = 1;
        pd_control_mask_(20) = 1;
        pd_control_mask_(21) = 1;
        pd_control_mask_(22) = 1;
        //////////////////////
        /////RIFHT ARM////////-0.3 -0.3 -1.5 1.27 1 0 1 0
        motion_q_(25) = -0.3;
        motion_q_(26) = -0.12;
        motion_q_(27) = -1.43;
        motion_q_(28) = 0.85;
        motion_q_(29) = 0.45; // elbow
        motion_q_(30) = -1;
        motion_q_(31) = 0.0;
        motion_q_(32) = 0.0;
        pd_control_mask_(25) = 1;
        pd_control_mask_(26) = 1;
        pd_control_mask_(27) = 1;
        pd_control_mask_(28) = 1;
        pd_control_mask_(29) = 1;
        pd_control_mask_(30) = 1;
        pd_control_mask_(31) = 1;
        pd_control_mask_(32) = 1;
        /////////////////////////////////////////////////////

        for (int i = 12; i < MODEL_DOF; i++)
        {
            motion_q_(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + 4, upperbody_mode_q_init_(i), 0, 0, motion_q_(i), 0, 0)(0);
        }
    }
    else if (upper_body_mode_ == 3) // Freezing
    {
        if (upperbody_mode_recieved_ == true)
        {
            cout << "Upperbody Mode is Changed to #3" << endl;
            cout << "----------Robot is Freezed---------" << endl;

            upperbody_mode_recieved_ = false;
            upperbody_mode_q_init_ = motion_q_pre_;

            std_msgs::String msg;
            std::stringstream upperbody_mode_ss;
            upperbody_mode_ss << "Robot is Freezed!";
            msg.data = upperbody_mode_ss.str();
            calibration_state_pub.publish(msg);
            calibration_state_gui_log_pub.publish(msg);
        }

        for (int i = 12; i < MODEL_DOF; i++)
        {
            motion_q_(i) = upperbody_mode_q_init_(i);
            pd_control_mask_(i) = 1;
        }
    }
    else if (upper_body_mode_ == 4) // READY pose
    {
        if (upperbody_mode_recieved_ == true)
        {
            cout << "Upperbody Mode is Changed to #4 (READY POSE)" << endl;
            upperbody_mode_recieved_ = false;
            upperbody_command_time_ = current_time_;
            upperbody_mode_q_init_ = motion_q_pre_;

            std_msgs::String msg;
            std::stringstream upperbody_mode_ss;
            upperbody_mode_ss << "Ready Pose is On!";
            msg.data = upperbody_mode_ss.str();
            calibration_state_pub.publish(msg);
            calibration_state_gui_log_pub.publish(msg);
        }
        ///////////////////////WAIST/////////////////////////
        motion_q_(12) = 0; // pitch
        motion_q_(13) = 0; // pitch
        motion_q_(14) = 0; // roll
        pd_control_mask_(12) = 1;
        pd_control_mask_(13) = 1;
        pd_control_mask_(14) = 1;
        /////////////////////////////////////////////////////

        ///////////////////////HEAD/////////////////////////
        motion_q_(23) = 0; // yaw
        motion_q_(24) = 0.3; // pitch
        pd_control_mask_(23) = 1;
        pd_control_mask_(24) = 1;
        /////////////////////////////////////////////////////

        ///////////////////////ARM/////////////////////////
        //////LEFT ARM///////0.3 0.3 1.5 -1.27 -1 0 -1 0
        motion_q_(15) = 0.0;
        motion_q_(16) = -0.3;
        motion_q_(17) = 1.57;
        motion_q_(18) = -1.2;
        motion_q_(19) = -1.57; // elbow
        motion_q_(20) = 1.5;
        motion_q_(21) = 0.4;
        motion_q_(22) = -0.2;
        pd_control_mask_(15) = 1;
        pd_control_mask_(16) = 1;
        pd_control_mask_(17) = 1;
        pd_control_mask_(18) = 1;
        pd_control_mask_(19) = 1;
        pd_control_mask_(20) = 1;
        pd_control_mask_(21) = 1;
        pd_control_mask_(22) = 1;
        //////////////////////
        /////RIFHT ARM////////-0.3 -0.3 -1.5 1.27 1 0 1 0
        motion_q_(25) = 0.0;
        motion_q_(26) = 0.3;
        motion_q_(27) = -1.57;
        motion_q_(28) = 1.2;
        motion_q_(29) = 1.57; // elbow
        motion_q_(30) = -1.5;
        motion_q_(31) = -0.4;
        motion_q_(32) = 0.2;
        pd_control_mask_(25) = 1;
        pd_control_mask_(26) = 1;
        pd_control_mask_(27) = 1;
        pd_control_mask_(28) = 1;
        pd_control_mask_(29) = 1;
        pd_control_mask_(30) = 1;
        pd_control_mask_(31) = 1;
        pd_control_mask_(32) = 1;
        /////////////////////////////////////////////////////

        for (int i = 12; i < MODEL_DOF; i++)
        {
            motion_q_(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + 4, upperbody_mode_q_init_(i), 0, 0, motion_q_(i), 0, 0)(0);
        }
    }
    else if (upper_body_mode_ == 5) // HEAD ONLY
    {
        if (still_pose_cali_flag_ == false)
        {
            cout << cred << " WARNING: Calibration is not completed! Upperbody returns to the init pose" << creset << endl;
            avatarUpperbodyModeUpdate(3);
            motion_q_ = motion_q_pre_;
        }
        else
        {
            if (upperbody_mode_recieved_ == true)
            {
                cout << "Upperbody Mode is Changed to #5 (HEAD ONLY MODE)" << endl;

                upperbody_mode_q_init_ = motion_q_pre_;
                first_loop_qp_retargeting_ = true;

                std_msgs::String msg;
                std::stringstream upperbody_mode_ss;
                upperbody_mode_ss << "HEAD Only Tracking Contorol in On";
                msg.data = upperbody_mode_ss.str();
                calibration_state_pub.publish(msg);
                calibration_state_gui_log_pub.publish(msg);
            }

            for (int i = 12; i < MODEL_DOF; i++)
            {
                motion_q_(i) = upperbody_mode_q_init_(i);
                pd_control_mask_(i) = 1;
            }

            rawMasterPoseProcessing();
            ///////////////////////HEAD/////////////////////////
            Vector3d error_w_head = -DyrosMath::getPhi(head_transform_pre_desired_from_.linear(), master_head_pose_.linear());
            error_w_head = head_transform_pre_desired_from_.linear().transpose() * error_w_head;
            error_w_head(0) = 0;
            error_w_head = head_transform_pre_desired_from_.linear() * error_w_head;

            MatrixXd J_temp, J_head, I3, J_inv_head;

            Vector3d u_dot_head = 200 * error_w_head;
            J_temp.setZero(6, MODEL_DOF_VIRTUAL);
            J_head.setZero(3, 2);
            I3.setIdentity(3, 3);

            RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, rd_.link_[Head].id, Eigen::Vector3d::Zero(), J_temp, false);
            J_head.block(0, 0, 3, 2) = J_temp.block(0, 29, 3, 2); // orientation
            J_inv_head = J_head.transpose() * (J_head * J_head.transpose() + I3 * 0.000001).inverse();

            for (int i = 0; i < 3; i++)
            {
                u_dot_head(i) = DyrosMath::minmax_cut(u_dot_head(i), -2.0, 2.0);
            }

            motion_q_dot_.segment(23, 2) = J_inv_head * u_dot_head;
            motion_q_dot_(23) = DyrosMath::minmax_cut(motion_q_dot_(23), joint_vel_limit_l_(23), joint_vel_limit_h_(23));
            motion_q_dot_(24) = DyrosMath::minmax_cut(motion_q_dot_(24), joint_vel_limit_l_(24), joint_vel_limit_h_(24));

            motion_q_.segment(23, 2) = motion_q_pre_.segment(23, 2) + motion_q_dot_.segment(23, 2) * dt_;
            motion_q_(23) = DyrosMath::minmax_cut(motion_q_(23), joint_limit_l_(23), joint_limit_h_(23));
            motion_q_(24) = DyrosMath::minmax_cut(motion_q_(24), joint_limit_l_(24), joint_limit_h_(24));

            // cout<<"master_head_pose_: \n"<<master_head_pose_.linear()<<endl;
            // motion_q_dot_.setZero();
        }
    }
    else if ((upper_body_mode_ == 6)) // HQPIK ver1
    {
        if (hmd_check_pose_calibration_[3] == false)
        {
            cout << cred << " WARNING: Calibration is not completed! Upperbody returns to the init pose" << creset << endl;
            avatarUpperbodyModeUpdate(3);
            motion_q_ = motion_q_pre_;
        }
        else
        {
            if (upperbody_mode_recieved_ == true)
            {
                cout << "Upperbody Mode is Changed to #6 (HQPIK1-AVATAR XPRIZE SEMIFINALS VERSION)" << endl;

                first_loop_hqpik_ = true;
                first_loop_qp_retargeting_ = true;

                std_msgs::String msg;
                std::stringstream upperbody_mode_ss;
                upperbody_mode_ss << "Motion Tracking Contorol in On (HQPIK1-AVATAR XPRIZE SEMIFINALS VERSION)";
                msg.data = upperbody_mode_ss.str();
                calibration_state_pub.publish(msg);
                calibration_state_gui_log_pub.publish(msg);
            }

            rawMasterPoseProcessing();
            motionRetargeting_HQPIK();
            // motionRetargeting_HQPIK_lexls();
            // motionRetargeting_QPIK_upperbody();
            // if (int(current_time_ * 10000) % 10000 == 0)
            // {
            //     cout<<"hqpik_time: "<< std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<endl;
            // }
        }
    }
    else if (upper_body_mode_ == 7) // HQPIK ver2
    {
        if (hmd_check_pose_calibration_[3] == false)
        {
            cout << cred << " WARNING: Calibration is not completed! Upperbody returns to the init pose" << creset << endl;
            avatarUpperbodyModeUpdate(3);
            motion_q_ = motion_q_pre_;
        }
        else
        {
            if (upperbody_mode_recieved_ == true)
            {
                cout << "Upperbody Mode is Changed to #7 (HQPIK ver2)" << endl;

                first_loop_hqpik2_ = true;
                first_loop_qp_retargeting_ = true;

                std_msgs::String msg;
                std::stringstream upperbody_mode_ss;
                upperbody_mode_ss << "Motion Tracking Contorol in On (HQPIK ver2)";
                msg.data = upperbody_mode_ss.str();
                calibration_state_pub.publish(msg);
                calibration_state_gui_log_pub.publish(msg);
            }

            rawMasterPoseProcessing();
            motionRetargeting_HQPIK();

            // if (int(current_time_ * 10000) % 10000 == 0)
            // {
            //     cout<<"hqpik_time: "<< std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<endl;
            // }
        }
    }
    else if (upper_body_mode_ == 8) // Absolute mapping
    {
        if (hmd_check_pose_calibration_[3] == false)
        {
            cout << cred << " WARNING: Calibration is not completed! Upperbody returns to the init pose" << creset << endl;
            avatarUpperbodyModeUpdate(3);
            motion_q_ = motion_q_pre_;
        }
        else
        {
            if (upperbody_mode_recieved_ == true)
            {
                cout << "Upperbody Mode is Changed to #8 (ABSOLUTE HAND POS MAPPING)" << endl;

                first_loop_hqpik_ = true;
                first_loop_qp_retargeting_ = true;

                std_msgs::String msg;
                std::stringstream upperbody_mode_ss;
                upperbody_mode_ss << "Upperbody Mode is Changed to #8 (ABSOLUTE HAND POS MAPPING)";
                msg.data = upperbody_mode_ss.str();
                calibration_state_pub.publish(msg);
                calibration_state_gui_log_pub.publish(msg);
            }

            rawMasterPoseProcessing();
            motionRetargeting_HQPIK();
            // motionRetargeting_HQPIK_lexls();
            // motionRetargeting_QPIK_upperbody();
            // if (int(current_time_ * 10000) % 10000 == 0)
            // {
            //     cout<<"hqpik_time: "<< std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<endl;
            // }
        }
    }
    else if (upper_body_mode_ == 9) // Propositional mapping
    {
        if (hmd_check_pose_calibration_[3] == false)
        {
            cout << cred << " WARNING: Calibration is not completed! Upperbody returns to the init pose" << creset << endl;
            avatarUpperbodyModeUpdate(3);
            motion_q_ = motion_q_pre_;
        }
        else
        {
            if (upperbody_mode_recieved_ == true)
            {
                cout << "Upperbody Mode is Changed to #9 (PROPOSITIONAL HAND POS MAPPING)" << endl;

                first_loop_hqpik_ = true;
                first_loop_qp_retargeting_ = true;

                std_msgs::String msg;
                std::stringstream upperbody_mode_ss;
                upperbody_mode_ss << "Upperbody Mode is Changed to #9 (PROPOSITIONAL HAND POS MAPPING)";
                msg.data = upperbody_mode_ss.str();
                calibration_state_pub.publish(msg);
                calibration_state_gui_log_pub.publish(msg);
            }

            rawMasterPoseProcessing();
            motionRetargeting_HQPIK();
            // motionRetargeting_HQPIK_lexls();
            // motionRetargeting_QPIK_upperbody();
            // if (int(current_time_ * 10000) % 10000 == 0)
            // {
            //     cout<<"hqpik_time: "<< std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<endl;
            // }
        }
    }
    else if (upper_body_mode_ == 10) // Cali Pose Direction Only
    {
        if (hmd_check_pose_calibration_[3] == false)
        {
            cout << cred << " WARNING: Calibration is not completed! Upperbody returns to the init pose" << creset << endl;
            avatarUpperbodyModeUpdate(3);
            motion_q_ = motion_q_pre_;
        }
        else
        {
            if (upperbody_mode_recieved_ == true)
            {
                cout << "Upperbody Mode is Changed to #10 (3D Mouse Mode)" << endl;

                first_loop_hqpik_ = true;
                first_loop_qp_retargeting_ = true;

                std_msgs::String msg;
                std::stringstream upperbody_mode_ss;
                upperbody_mode_ss << "Upperbody Mode is Changed to #10 (3D Mouse Mode)";
                msg.data = upperbody_mode_ss.str();
                calibration_state_pub.publish(msg);
                calibration_state_gui_log_pub.publish(msg);
            }

            rawMasterPoseProcessing();
            motionRetargeting_HQPIK();
            // motionRetargeting_HQPIK_lexls();
            // motionRetargeting_QPIK_upperbody();
            // if (int(current_time_ * 10000) % 10000 == 0)
            // {
            //     cout<<"hqpik_time: "<< std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() <<endl;
            // }
        }
    }

}

void AvatarController::motionRetargeting_HQPIK()
{
    // const unsigned int hierarchy_num_hqpik_ = 3;
    // const unsigned int variable_size_hqpik_ = 21;
	// const unsigned int constraint_size1_hqpik_ = 21;	//[lb <=	x	<= 	ub] form constraints
	// const unsigned int constraint_size2_hqpik_[3] = {12, 15, 17};	//[lb <=	Ax 	<=	ub] or [Ax = b]
	// const unsigned int control_size_hqpik_[3] = {3, 14, 8};		//1: upperbody, 2: head + hand, 3: upperarm + shoulder AAC

    if (first_loop_hqpik_)
    {
        for (int i = 0; i < hierarchy_num_hqpik_; i++)
        {
            QP_qdot_hqpik_.resize(hierarchy_num_hqpik_);
            QP_qdot_hqpik_[i].InitializeProblemSize(variable_size_hqpik_, constraint_size2_hqpik_[i]);
            J_hqpik_[i].setZero(control_size_hqpik_[i], variable_size_hqpik_);
            u_dot_hqpik_[i].setZero(control_size_hqpik_[i]);

            ubA_hqpik_[i].setZero(constraint_size2_hqpik_[i]);
            lbA_hqpik_[i].setZero(constraint_size2_hqpik_[i]);

            H_hqpik_[i].setZero(variable_size_hqpik_, variable_size_hqpik_);
            g_hqpik_[i].setZero(variable_size_hqpik_);

            ub_hqpik_[i].setZero(constraint_size1_hqpik_);
            lb_hqpik_[i].setZero(constraint_size1_hqpik_);

            q_dot_hqpik_[i].setZero(variable_size_hqpik_);

            w1_hqpik_[i] = 2500;  // upperbody tracking (2500)
            w2_hqpik_[i] = 50;    // kinetic energy (50)
            w3_hqpik_[i] = 0.000; // acceleration (0.000)
        }

        // upper arm & shoulder orientation control gain
        w1_hqpik_[2] = 250;   // upperbody tracking (250)
        w2_hqpik_[2] = 50;    // kinetic energy (50)
        w3_hqpik_[2] = 0.001; // acceleration (0.002)

        // shoulder orientation control gain
        // w1_hqpik_[3] = 250;   // upperbody tracking (250)
        // w2_hqpik_[3] = 50;    // kinetic energy (50)
        // w3_hqpik_[3] = 0.001; // acceleration (0.002)

        last_solved_hierarchy_num_ = 2;

        first_loop_hqpik_ = false;
    }

    // VectorQVQd q_desired_pre;
    // q_desired_pre.setZero();
    // q_desired_pre(39) = 1;
    // q_desired_pre.segment(6, MODEL_DOF) = pre_desired_q_;
    Vector3d zero3;
    zero3.setZero();
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);

    ////1st Task
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, link_avatar_[Upper_Body].id, zero3, J_temp_, true);
    J_hqpik_[0].block(0, 0, 3, variable_size_hqpik_) = J_temp_.block(0, 18, 3, variable_size_hqpik_); // orientation

    Vector3d error_w_upperbody = -DyrosMath::getPhi(upperbody_transform_pre_desired_from_.linear(), master_upperbody_pose_.linear());
    u_dot_hqpik_[0] = 100 * error_w_upperbody;

    ////2nd Task
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand].id, lhand_control_point_offset_, J_temp_, false);
    J_hqpik_[1].block(0, 0, 3, variable_size_hqpik_) = J_temp_.block(3, 18, 3, variable_size_hqpik_); // position
    J_hqpik_[1].block(3, 0, 3, variable_size_hqpik_) = J_temp_.block(0, 18, 3, variable_size_hqpik_); // orientation
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand].id, rhand_control_point_offset_, J_temp_, false);
    J_hqpik_[1].block(6, 0, 3, variable_size_hqpik_) = J_temp_.block(3, 18, 3, variable_size_hqpik_); // position
    J_hqpik_[1].block(9, 0, 3, variable_size_hqpik_) = J_temp_.block(0, 18, 3, variable_size_hqpik_); // orientation
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, link_avatar_[Head].id, zero3, J_temp_, false);
    J_hqpik_[1].block(12, 0, 2, variable_size_hqpik_) = (head_transform_pre_desired_from_.linear().transpose() * J_temp_.block(0, 18, 3, variable_size_hqpik_)).block(1, 0, 2, variable_size_hqpik_); // orientation

    // Hand error
    Vector3d error_v_lhand = master_lhand_pose_.translation() - lhand_transform_pre_desired_from_.translation();
    Vector3d error_w_lhand = -DyrosMath::getPhi(lhand_transform_pre_desired_from_.linear(), master_lhand_pose_.linear());
    Vector3d error_v_rhand = master_rhand_pose_.translation() - rhand_transform_pre_desired_from_.translation();
    Vector3d error_w_rhand = -DyrosMath::getPhi(rhand_transform_pre_desired_from_.linear(), master_rhand_pose_.linear());

    // Head error
    Vector3d error_w_head = -DyrosMath::getPhi(head_transform_pre_desired_from_.linear(), master_head_pose_.linear());
    error_w_head = head_transform_pre_desired_from_.linear().transpose() * error_w_head;
    error_w_head(0) = 0;

    u_dot_hqpik_[1].segment(0, 3) = 200 * error_v_lhand;
    u_dot_hqpik_[1].segment(3, 3) = 100 * error_w_lhand;
    u_dot_hqpik_[1].segment(6, 3) = 200 * error_v_rhand;
    u_dot_hqpik_[1].segment(9, 3) = 100 * error_w_rhand;
    u_dot_hqpik_[1].segment(12, 2) = 200 * error_w_head.segment(1, 2);

    ////3rd Task
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, link_avatar_[Left_Hand - 4].id, zero3, J_temp_, false);
    J_hqpik_[2].block(0, 0, 2, variable_size_hqpik_) = (lupperarm_transform_pre_desired_from_.linear().transpose() * J_temp_.block(0, 18, 3, variable_size_hqpik_)).block(1, 0, 2, variable_size_hqpik_); // orientation
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);

    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, link_avatar_[Right_Hand - 4].id, zero3, J_temp_, false);
    J_hqpik_[2].block(2, 0, 2, variable_size_hqpik_) = (rupperarm_transform_pre_desired_from_.linear().transpose() * J_temp_.block(0, 18, 3, variable_size_hqpik_)).block(1, 0, 2, variable_size_hqpik_); // orientation

    // Upperarm error
    Vector3d error_w_lupperarm = -DyrosMath::getPhi(lupperarm_transform_pre_desired_from_.linear(), master_lelbow_pose_.linear());
    error_w_lupperarm = lupperarm_transform_pre_desired_from_.linear().transpose() * error_w_lupperarm;
    error_w_lupperarm(0) = 0;

    Vector3d error_w_rupperarm = -DyrosMath::getPhi(rupperarm_transform_pre_desired_from_.linear(), master_relbow_pose_.linear());
    error_w_rupperarm = rupperarm_transform_pre_desired_from_.linear().transpose() * error_w_rupperarm;
    error_w_rupperarm(0) = 0;

    u_dot_hqpik_[2].segment(0, 2) = 100 * error_w_lupperarm.segment(1, 2);
    u_dot_hqpik_[2].segment(2, 2) = 100 * error_w_rupperarm.segment(1, 2);


    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, rd_.link_[Left_Hand - 6].id, zero3, J_temp_, false);
    J_hqpik_[2].block(4, 0, 2, variable_size_hqpik_) = (lacromion_transform_pre_desired_from_.linear().transpose() * J_temp_.block(0, 18, 3, variable_size_hqpik_)).block(1, 0, 2, variable_size_hqpik_); // orientation
    J_temp_.setZero(6, MODEL_DOF_VIRTUAL);
    RigidBodyDynamics::CalcPointJacobian6D(model_d_, pre_desired_q_qvqd_, rd_.link_[Right_Hand - 6].id, zero3, J_temp_, false);
    J_hqpik_[2].block(6, 0, 2, variable_size_hqpik_) = (racromion_transform_pre_desired_from_.linear().transpose() * J_temp_.block(0, 18, 3, variable_size_hqpik_)).block(1, 0, 2, variable_size_hqpik_); // orientation

    // Shoulder error
    Vector3d error_w_lshoulder = -DyrosMath::getPhi(lacromion_transform_pre_desired_from_.linear(), master_lshoulder_pose_.linear());
    error_w_lshoulder = lacromion_transform_pre_desired_from_.linear().transpose() * error_w_lshoulder;
    error_w_lshoulder(0) = 0;

    Vector3d error_w_rshoulder = -DyrosMath::getPhi(racromion_transform_pre_desired_from_.linear(), master_rshoulder_pose_.linear());
    error_w_rshoulder = racromion_transform_pre_desired_from_.linear().transpose() * error_w_rshoulder;
    error_w_rshoulder(0) = 0;

    u_dot_hqpik_[2].segment(4, 2) = 100 * error_w_lshoulder.segment(1, 2);
    u_dot_hqpik_[2].segment(6, 2) = 100 * error_w_rshoulder.segment(1, 2);

    for (int i = 0; i < hierarchy_num_hqpik_; i++)
    {
        if (i > last_solved_hierarchy_num_)
        {
            QP_qdot_hqpik_[i].InitializeProblemSize(variable_size_hqpik_, constraint_size2_hqpik_[i]);
        }

        if(last_solved_hierarchy_num_ == 0)
        {
            QP_qdot_hqpik_[0].InitializeProblemSize(variable_size_hqpik_, constraint_size2_hqpik_[0]);
        }
    }

    last_solved_hierarchy_num_ = -1;

    for (int i = 0; i < hierarchy_num_hqpik_; i++)
    {
        MatrixXd H1, H2, H3;
        VectorXd g1, g2, g3;

        H1 = J_hqpik_[i].transpose() * J_hqpik_[i];
        H2 = Eigen::MatrixXd::Identity(variable_size_hqpik_, variable_size_hqpik_);
        // H2 = A_mat_.block(18, 18, variable_size_hqpik_, variable_size_hqpik_) + Eigen::MatrixXd::Identity(variable_size_hqpik_, variable_size_hqpik_) * (2e-2);
        H2(3, 3) += 10;   // left arm 1st joint
        H2(13, 13) += 10; // right arm 1st joint
        H3 = Eigen::MatrixXd::Identity(variable_size_hqpik_, variable_size_hqpik_) * (2000) * (2000);

        g1 = -J_hqpik_[i].transpose() * u_dot_hqpik_[i];
        g2.setZero(variable_size_hqpik_);
        g3 = -motion_q_dot_pre_.segment(12, variable_size_hqpik_) * (2000) * (2000);

        if (i >= 2)
        {
        }

        H_hqpik_[i] = w1_hqpik_[i] * H1 + w2_hqpik_[i] * H2 + w3_hqpik_[i] * H3;
        g_hqpik_[i] = w1_hqpik_[i] * g1 + w2_hqpik_[i] * g2 + w3_hqpik_[i] * g3;

        double speed_reduce_rate = 20; // when the current joint position is near joint limit (10 degree), joint limit condition is activated.

        for (int j = 0; j < constraint_size1_hqpik_; j++)
        {
            lb_hqpik_[i](j) = min(max(speed_reduce_rate * (joint_limit_l_(j + 12) - motion_q_pre_(j + 12)), joint_vel_limit_l_(j + 12)), joint_vel_limit_h_(j + 12));
            ub_hqpik_[i](j) = max(min(speed_reduce_rate * (joint_limit_h_(j + 12) - motion_q_pre_(j + 12)), joint_vel_limit_h_(j + 12)), joint_vel_limit_l_(j + 12));
        }

        A_hqpik_[i].setZero(constraint_size2_hqpik_[i], variable_size_hqpik_);

        int higher_task_equality_num = 0;
        for (int h = 0; h < i; h++)
        {
            A_hqpik_[i].block(higher_task_equality_num, 0, control_size_hqpik_[h], variable_size_hqpik_) = J_hqpik_[h];
            // ubA_hqpik_[i].segment(higher_task_equality_num, control_size_hqpik_[h]) = J_hqpik_[h] * q_dot_hqpik_[h];
            // lbA_hqpik_[i].segment(higher_task_equality_num, control_size_hqpik_[h]) = J_hqpik_[h] * q_dot_hqpik_[h];
            ubA_hqpik_[i].segment(higher_task_equality_num, control_size_hqpik_[h]) = J_hqpik_[h] * q_dot_hqpik_[i - 1];
            lbA_hqpik_[i].segment(higher_task_equality_num, control_size_hqpik_[h]) = J_hqpik_[h] * q_dot_hqpik_[i - 1];
            higher_task_equality_num += control_size_hqpik_[h];
        }

        // hand velocity constraints
        if (i < 2)
        {
            A_hqpik_[i].block(higher_task_equality_num, 0, 12, variable_size_hqpik_) = J_hqpik_[1].block(0, 0, 12, variable_size_hqpik_);

            for (int j = 0; j < 3; j++)
            {
                // linear velocity limit
                lbA_hqpik_[i](higher_task_equality_num + j) = -0.8;
                ubA_hqpik_[i](higher_task_equality_num + j) = 0.8;
                lbA_hqpik_[i](higher_task_equality_num + j + 6) = -0.8;
                ubA_hqpik_[i](higher_task_equality_num + j + 6) = 0.8;

                // angular velocity limit
                lbA_hqpik_[i](higher_task_equality_num + j + 3) = -2*M_PI;
                ubA_hqpik_[i](higher_task_equality_num + j + 3) = 2*M_PI;
                lbA_hqpik_[i](higher_task_equality_num + j + 9) = -2*M_PI;
                ubA_hqpik_[i](higher_task_equality_num + j + 9) = 2*M_PI;
            }
        }

        // QP_qdot_hqpik_[i].SetPrintLevel(PL_NONE);
        QP_qdot_hqpik_[i].EnableEqualityCondition(equality_condition_eps_);
        QP_qdot_hqpik_[i].UpdateMinProblem(H_hqpik_[i], g_hqpik_[i]);
        QP_qdot_hqpik_[i].UpdateSubjectToAx(A_hqpik_[i], lbA_hqpik_[i], ubA_hqpik_[i]);
        QP_qdot_hqpik_[i].UpdateSubjectToX(lb_hqpik_[i], ub_hqpik_[i]);

        // cout<<"test8"<<endl;
        if (QP_qdot_hqpik_[i].SolveQPoases(200, qpres_hqpik_))
        {
            q_dot_hqpik_[i] = qpres_hqpik_.segment(0, variable_size_hqpik_);

            last_solved_hierarchy_num_ = i;

            // if(i == 3)
            // {
            //     if (int(current_time_ * 10000) % 1000 == 0)
            //         std::cout << "4th HQPIK(shoulder) is solved" << std::endl;
            // }
        }
        else
        {
            q_dot_hqpik_[i].setZero();

            // last_solved_hierarchy_num_ = max(i-1, 0);

            last_solved_hierarchy_num_ = DyrosMath::minmax_cut(last_solved_hierarchy_num_, 0, hierarchy_num_hqpik_-1);
            
            if (i == 0)
            {
                if(true)
                {
                    std::cout << "Error hierarchy: " << 0 << std::endl;
                    std::cout << "last solved q_dot: " << q_dot_hqpik_[0].transpose() << std::endl;
                }
            }
            else
            {
                if(last_solved_hierarchy_num_ < 0)
                {
                    std::cout<<"last_solved_hierarchy_num_ is negative!! "<< std::endl;
                }
                // if (int(current_time_ * 2000) % 1000 == 0)
                if(true)
                {
                    std::cout << "Error hierarchy: " << i << std::endl;
                    std::cout << "last solved q_dot: " << q_dot_hqpik_[last_solved_hierarchy_num_].transpose() << std::endl;
                }
            }
            // cout<<"Error qpres_: \n"<< qpres_ << endl;
            break;
        }
    }
    
    if(last_solved_hierarchy_num_ < 0)
    {
        std::cout<<"last_solved_hierarchy_num_ is negative!! "<< std::endl;
    }

    for (int i = 0; i < variable_size_hqpik_; i++)
    {
        motion_q_dot_(12 + i) = q_dot_hqpik_[last_solved_hierarchy_num_](i);
        motion_q_(12 + i) = motion_q_pre_(12 + i) + motion_q_dot_(12 + i) * dt_;
        pd_control_mask_(12 + i) = 1;
    }
}

//////////Self Collision Avoidance Network////////////////
void AvatarController::initializeScaMlp(MLP &mlp, int n_input, int n_output, Eigen::VectorXd n_hidden, Eigen::VectorXd q_to_input_mapping_vector)
{
    mlp.n_input = n_input;
    mlp.n_output = n_output;
    mlp.n_hidden = n_hidden;
    mlp.n_layer = n_hidden.rows()+1; // hiden layers + output layer
    mlp.q_to_input_mapping_vector = q_to_input_mapping_vector;

    mlp.weight.resize(mlp.n_layer);
    mlp.bias.resize(mlp.n_layer);
    mlp.hidden.resize(mlp.n_layer-1);
    mlp.hidden_derivative.resize(mlp.n_layer-1);

    mlp.w_path.resize(mlp.n_layer);
    mlp.b_path.resize(mlp.n_layer);

    mlp.weight_files.resize(mlp.n_layer);
    mlp.bias_files.resize(mlp.n_layer);
    //parameters resize
    for (int i = 0; i < mlp.n_layer; i++)
    {   
        
        if(i == 0)
        {
            mlp.weight[i].setZero(mlp.n_hidden(i), mlp.n_input);
            mlp.bias[i].setZero(mlp.n_hidden(i));
            mlp.hidden[i].setZero(mlp.n_hidden(i));
            mlp.hidden_derivative[i].setZero(mlp.n_hidden(i), mlp.n_input);
        }
        else if(i == mlp.n_layer - 1)
        {
            mlp.weight[i].setZero(mlp.n_output, mlp.n_hidden(i-1));
            mlp.bias[i].setZero(mlp.n_output);
        }
        else
        {
            mlp.weight[i].setZero(mlp.n_hidden(i), mlp.n_hidden(i-1));
            mlp.bias[i].setZero(mlp.n_hidden(i));
            mlp.hidden[i].setZero(mlp.n_hidden(i));
            mlp.hidden_derivative[i].setZero(mlp.n_hidden(i), mlp.n_hidden(i-1));
        }
    }

    //input output resize
    mlp.input_slow.setZero(mlp.n_input);
    mlp.input_fast.setZero(mlp.n_input);
    mlp.input_thread.setZero(mlp.n_input);

    mlp.output_slow.setZero(mlp.n_output);
    mlp.output_fast.setZero(mlp.n_output);
    mlp.output_thread.setZero(mlp.n_output);

    mlp.output_derivative_fast.setZero(mlp.n_output, mlp.n_input);
    mlp.hx_gradient_fast.setZero(mlp.n_input);
    mlp.hx_gradient_fast_lpf.setZero(mlp.n_input);
    mlp.hx_gradient_fast_pre.setZero(mlp.n_input);

    mlp.self_collision_stop_cnt_ = 0;
}
void AvatarController::loadScaNetwork(MLP &mlp, std::string folder_path)
{
    for(int i =0; i<mlp.n_layer; i++)
    {
        mlp.w_path[i] = folder_path + "weight_" + std::to_string(i) + ".txt";
        mlp.b_path[i] = folder_path + "bias_" + std::to_string(i) + ".txt";

        mlp.weight_files[i].open(mlp.w_path[i], ios::in);
        mlp.bias_files[i].open(mlp.b_path[i], ios::in);

        readWeightFile(mlp, i);
        readBiasFile(mlp, i);
    }
}
void AvatarController::readWeightFile(MLP &mlp, int weight_num)
{
    if (!mlp.weight_files[weight_num].is_open())
    {
        std::cout << "Can not find the file: " << mlp.w_path[weight_num] << std::endl;
    }
    for(int i = 0; i<mlp.weight[weight_num].rows() ; i++)
    {
        for(int j = 0; j<mlp.weight[weight_num].cols() ; j++)
        {
            mlp.weight_files[weight_num] >> mlp.weight[weight_num](i, j);
        }
    }
    mlp.weight_files[weight_num].close();

    if(mlp.loadweightfile_verbose == true)
    {
        cout<<"weight_"<<weight_num<<": \n"<< mlp.weight[weight_num] <<endl;
    }
}
void AvatarController::readBiasFile(MLP &mlp, int bias_num)
{
    if (!mlp.bias_files[bias_num].is_open())
    {
        std::cout << "Can not find the file: " << mlp.b_path[bias_num] << std::endl;
    }
    for(int i = 0; i<mlp.bias[bias_num].rows() ; i++)
    {
        mlp.bias_files[bias_num] >> mlp.bias[bias_num](i);
    }
    mlp.bias_files[bias_num].close();

    if(mlp.loadbiasfile_verbose == true)
    {
        cout<<"bias_"<<bias_num - mlp.n_layer<< ": \n"<< mlp.bias[bias_num] <<endl;
    }
}

void AvatarController::calculateScaMlpOutput(MLP &mlp)
{
    if(atb_mlp_input_update_ == false)
    {
        atb_mlp_input_update_ = true;
        mlp.input_fast = mlp.input_thread;
        q_ddot_max_fast_ = q_ddot_max_thread_;
        atb_mlp_input_update_ = false;
    }
    MatrixXd temp_derivative_pi; 
    for(int layer = 0; layer < mlp.n_layer; layer++)
    {
        if(layer == 0)  // input layer
        {
            mlp.hidden[0] = mlp.weight[0]*mlp.input_fast + mlp.bias[0];
            for(int h=0; h<mlp.n_hidden(layer); h++)
            {
                mlp.hidden[0](h) = std::tanh(mlp.hidden[0](h));   //activation function
                mlp.hidden_derivative[0].row(h) = (1-(mlp.hidden[0](h)*mlp.hidden[0](h)))*mlp.weight[0].row(h); //derivative wrt input
            }
            temp_derivative_pi = mlp.hidden_derivative[0];
        }
        else if(layer == mlp.n_layer - 1)   // output layer
        {
            mlp.output_fast = mlp.weight[layer]*mlp.hidden[layer-1] + mlp.bias[layer];
            mlp.output_derivative_fast = mlp.weight[layer]*temp_derivative_pi;
        }
        else    // hidden layers
        {
            mlp.hidden[layer] = mlp.weight[layer]*mlp.hidden[layer-1] + mlp.bias[layer];
            for(int h=0; h<mlp.n_hidden(layer); h++)
            {
                mlp.hidden[layer](h) = std::tanh(mlp.hidden[layer](h));   //activation function
                mlp.hidden_derivative[layer].row(h) = (1-(mlp.hidden[layer](h)*mlp.hidden[layer](h)))*mlp.weight[layer].row(h); //derivative wrt input
            }
            temp_derivative_pi =  mlp.hidden_derivative[layer]*temp_derivative_pi;
        }
    }

    if(sca_dynamic_version_)
    {
        // for(int i=0; i<mlp.n_input; i++)
        // {
            // if(q_ddot_max_fast_(mlp.q_to_input_mapping_vector(i)) !=0 )
            // {
            //     mlp.output_derivative_fast.col(i) = mlp.output_derivative_fast.col(i)*
            //     ( 1 - desired_q_ddot_(mlp.q_to_input_mapping_vector(i))/q_ddot_max_fast_(mlp.q_to_input_mapping_vector(i)) );
            // }
        // }
    }

    mlp.hx_gradient_fast_pre = mlp.hx_gradient_fast;
    mlp.hx_gradient_fast = (mlp.output_derivative_fast.row(1) - mlp.output_derivative_fast.row(0)).transpose();
    for(int i=0; i<mlp.n_input; i++)
    {
        mlp.hx_gradient_fast_lpf(i) = DyrosMath::lpf(mlp.hx_gradient_fast(i), mlp.hx_gradient_fast_pre(i), 1/dt_, 10.0);
    }

    mlp.hx = mlp.output_fast(1) - mlp.output_fast(0);


    // if(atb_mlp_output_update_ == false)
    // {
    //     atb_mlp_output_update_ = true;
    //     mlp.output_thread = mlp.output_fast;
    //     atb_mlp_output_update_ = false;
    // }
}
////////////////////////////////////////////////////////////////////////////////////////////
void AvatarController::poseCalibration()
{
    hmd_tracker_status_ = hmd_tracker_status_raw_;

    if (hmd_tracker_status_ == true)
    {
        if (hmd_tracker_status_pre_ == false)
        {
            tracker_status_changed_time_ = current_time_;
            cout << "tracker is attatched" << endl;

            std_msgs::String msg;
            std::stringstream upperbody_mode_ss;
            upperbody_mode_ss << "tracker is attatched";
            msg.data = upperbody_mode_ss.str();
            calibration_state_pub.publish(msg);
            calibration_state_gui_log_pub.publish(msg);
        }

        hmd_head_pose_ = hmd_head_pose_raw_;
        hmd_lshoulder_pose_ = hmd_lshoulder_pose_raw_;
        hmd_lupperarm_pose_ = hmd_lupperarm_pose_raw_;
        hmd_lhand_pose_ = hmd_lhand_pose_raw_;
        hmd_rshoulder_pose_ = hmd_rshoulder_pose_raw_;
        hmd_rupperarm_pose_ = hmd_rupperarm_pose_raw_;
        hmd_rhand_pose_ = hmd_rhand_pose_raw_;
        hmd_chest_pose_ = hmd_chest_pose_raw_;
        hmd_pelv_pose_ = hmd_pelv_pose_raw_;

        if (current_time_ - tracker_status_changed_time_ <= 3)
        {
            // double w = DyrosMath::cubic(current_time_, tracker_status_changed_time_, tracker_status_changed_time_+5, 0, 1, 0, 0);
            double w = (current_time_ - tracker_status_changed_time_) / 3;
            w = DyrosMath::minmax_cut(w, 0.0, 1.0);

            hmd_head_pose_.translation() = w * hmd_head_pose_raw_.translation() + (1 - w) * hmd_head_pose_raw_last_.translation();
            hmd_lupperarm_pose_.translation() = w * hmd_lupperarm_pose_raw_.translation() + (1 - w) * hmd_lupperarm_pose_raw_last_.translation();
            hmd_lhand_pose_.translation() = w * hmd_lhand_pose_raw_.translation() + (1 - w) * hmd_lhand_pose_raw_last_.translation();
            hmd_rupperarm_pose_.translation() = w * hmd_rupperarm_pose_raw_.translation() + (1 - w) * hmd_rupperarm_pose_raw_last_.translation();
            hmd_rhand_pose_.translation() = w * hmd_rhand_pose_raw_.translation() + (1 - w) * hmd_rhand_pose_raw_last_.translation();
            hmd_chest_pose_.translation() = w * hmd_chest_pose_raw_.translation() + (1 - w) * hmd_chest_pose_raw_last_.translation();
            hmd_pelv_pose_.translation() = w * hmd_pelv_pose_raw_.translation() + (1 - w) * hmd_pelv_pose_raw_last_.translation();

            Eigen::AngleAxisd head_ang_diff(hmd_head_pose_raw_.linear() * hmd_head_pose_raw_last_.linear().transpose());
            Eigen::AngleAxisd lelbow_ang_diff(hmd_lupperarm_pose_raw_.linear() * hmd_lupperarm_pose_raw_last_.linear().transpose());
            Eigen::AngleAxisd lhand_ang_diff(hmd_lhand_pose_raw_.linear() * hmd_lhand_pose_raw_last_.linear().transpose());
            Eigen::AngleAxisd relbow_ang_diff(hmd_rupperarm_pose_raw_.linear() * hmd_rupperarm_pose_raw_last_.linear().transpose());
            Eigen::AngleAxisd rhand_ang_diff(hmd_rhand_pose_raw_.linear() * hmd_rhand_pose_raw_last_.linear().transpose());
            Eigen::AngleAxisd upperbody_ang_diff(hmd_chest_pose_raw_.linear() * hmd_chest_pose_raw_last_.linear().transpose());
            Eigen::AngleAxisd pelv_ang_diff(hmd_pelv_pose_raw_.linear() * hmd_pelv_pose_raw_last_.linear().transpose());

            Eigen::Matrix3d lhand_diff_m, rhand_diff_m, lelbow_diff_m, relbow_diff_m, head_diff_m, upperbody_diff_m, pelv_diff_m;
            lhand_diff_m = Eigen::AngleAxisd(lhand_ang_diff.angle() * w, lhand_ang_diff.axis());
            rhand_diff_m = Eigen::AngleAxisd(rhand_ang_diff.angle() * w, rhand_ang_diff.axis());
            lelbow_diff_m = Eigen::AngleAxisd(lelbow_ang_diff.angle() * w, lelbow_ang_diff.axis());
            relbow_diff_m = Eigen::AngleAxisd(relbow_ang_diff.angle() * w, relbow_ang_diff.axis());
            head_diff_m = Eigen::AngleAxisd(head_ang_diff.angle() * w, head_ang_diff.axis());
            upperbody_diff_m = Eigen::AngleAxisd(upperbody_ang_diff.angle() * w, upperbody_ang_diff.axis());
            pelv_diff_m = Eigen::AngleAxisd(pelv_ang_diff.angle() * w, pelv_ang_diff.axis());

            hmd_lupperarm_pose_.linear() = lelbow_diff_m * hmd_lupperarm_pose_raw_last_.linear();
            hmd_lhand_pose_.linear() = lhand_diff_m * hmd_lhand_pose_raw_last_.linear();
            hmd_rupperarm_pose_.linear() = relbow_diff_m * hmd_rupperarm_pose_raw_last_.linear();
            hmd_rhand_pose_.linear() = rhand_diff_m * hmd_rhand_pose_raw_last_.linear();
            hmd_head_pose_.linear() = head_diff_m * hmd_head_pose_raw_last_.linear();
            hmd_chest_pose_.linear() = upperbody_diff_m * hmd_chest_pose_raw_last_.linear();
            hmd_pelv_pose_.linear() = pelv_diff_m * hmd_pelv_pose_raw_last_.linear();

            if (int((current_time_ - tracker_status_changed_time_) * 2000) % 1000 == 0)
                cout << "Motion Tracking Resume!" << int((current_time_ - tracker_status_changed_time_) / 3 * 100) << "%" << endl;
        }
        else
        {
        }
    }
    else // false
    {
        if (hmd_tracker_status_pre_ == true)
        {
            tracker_status_changed_time_ = current_time_;
            cout << cred << "tracker is detatched" << creset << endl;

            std_msgs::String msg;
            std::stringstream upperbody_mode_ss;
            upperbody_mode_ss << "tracker is detatched";
            msg.data = upperbody_mode_ss.str();
            calibration_state_pub.publish(msg);
            calibration_state_gui_log_pub.publish(msg);

            hmd_head_pose_raw_last_ = hmd_head_pose_raw_;
            hmd_lupperarm_pose_raw_last_ = hmd_lupperarm_pose_raw_;
            hmd_lhand_pose_raw_last_ = hmd_lhand_pose_raw_;
            hmd_rupperarm_pose_raw_last_ = hmd_rupperarm_pose_raw_;
            hmd_rhand_pose_raw_last_ = hmd_rhand_pose_raw_;
            hmd_chest_pose_raw_last_ = hmd_chest_pose_raw_;
            hmd_pelv_pose_raw_last_ = hmd_pelv_pose_raw_;
        }

        hmd_head_pose_ = hmd_head_pose_raw_last_;
        hmd_lupperarm_pose_ = hmd_lupperarm_pose_raw_last_;
        hmd_lhand_pose_ = hmd_lhand_pose_raw_last_;
        hmd_rupperarm_pose_ = hmd_rupperarm_pose_raw_last_;
        hmd_rhand_pose_ = hmd_rhand_pose_raw_last_;
        hmd_chest_pose_ = hmd_chest_pose_raw_last_;
        hmd_pelv_pose_ = hmd_pelv_pose_raw_last_;
    }


    hmd_pelv_vel_.segment(0, 3) = (hmd_pelv_pose_.translation() - hmd_pelv_pose_pre_.translation()) / dt_;
    Eigen::AngleAxisd ang_temp(hmd_pelv_pose_.linear() * hmd_pelv_pose_pre_.linear().transpose());
    hmd_pelv_vel_.segment(3, 3) = ang_temp.axis() * ang_temp.angle() / dt_;

    bool fast_pelv_move = false;
    bool far_pelv_move = false;
    int maximum_data_cut_num = 200;
    double pelv_max_vel = 5;
    double pelv_pos_boundary = 0.3;

    if ((hmd_check_pose_calibration_[3] == true) && (hmd_tracker_status_ == true) && (current_time_ - tracker_status_changed_time_ > 5))
    {
        // hmd_pelv_pose_ = velocityFilter(hmd_pelv_pose_, hmd_pelv_pose_pre_, hmd_pelv_vel_, pelv_max_vel, hmd_pelv_abrupt_motion_count_, maximum_data_cut_num, fast_pelv_move);

        // if (fast_pelv_move)
        // {
        //     cout << "Fast Pelvis Movement is Detected! (" << hmd_pelv_abrupt_motion_count_ << ")" << endl;
        // }

        // if (far_pelv_move)
        // {
        //     cout << "WARNING: The Operator Has Left The Init Position!! \n"
        //          << "Robot is stopped (upperbody mode #3)" << endl;
        //     upper_body_mode_ = 3;
        // }
    }

    Eigen::Vector3d hmd_pelv_rpy;
    Eigen::Matrix3d hmd_pelv_yaw_rot;
    Eigen::Isometry3d hmd_pelv_pose_yaw_only;

    if ((hmd_check_pose_calibration_[0] == true) && (still_pose_cali_flag_ == false))
    {
        hmd_pelv_pose_yaw_only.translation() = hmd_pelv_pose_.translation();
        hmd_pelv_rpy = DyrosMath::rot2Euler(hmd_pelv_pose_.linear());
        hmd_pelv_yaw_rot = DyrosMath::rotateWithZ(hmd_pelv_rpy(2));
        hmd_pelv_pose_yaw_only.linear() = hmd_pelv_yaw_rot;
    }
    else
    {
        hmd_pelv_pose_yaw_only.translation() = hmd_pelv_pose_init_.translation();
        hmd_pelv_rpy = DyrosMath::rot2Euler(hmd_pelv_pose_init_.linear());
        hmd_pelv_yaw_rot = DyrosMath::rotateWithZ(hmd_pelv_rpy(2));
        hmd_pelv_pose_yaw_only.linear() = hmd_pelv_yaw_rot;
    }

    // coordinate conversion
    if(master_arm_mode_ == false)
    {
        hmd_head_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_head_pose_;
        hmd_lupperarm_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_lupperarm_pose_;
        hmd_lhand_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_lhand_pose_;
        hmd_rupperarm_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_rupperarm_pose_;
        hmd_rhand_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_rhand_pose_;
        hmd_chest_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_chest_pose_;
        // hmd_pelv_pose_.linear().setIdentity();
    }
    else
    {
        hmd_head_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_head_pose_;
        hmd_lupperarm_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_lupperarm_pose_;
        // hmd_lhand_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_lhand_pose_;
        hmd_rupperarm_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_rupperarm_pose_;
        // hmd_rhand_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_rhand_pose_;
        hmd_chest_pose_ = hmd_pelv_pose_yaw_only.inverse() * hmd_chest_pose_;
        // orientation offset
        hmd_lhand_pose_.linear() = hmd_lhand_pose_.linear()*DyrosMath::rotateWithY(-90*DEG2RAD);
        hmd_rhand_pose_.linear() = hmd_rhand_pose_.linear()*DyrosMath::rotateWithY(-90*DEG2RAD);
    }

    Eigen::Vector3d tracker_offset;
    // tracker_offset << -0.08, 0, 0; // bebop
    tracker_offset << -0.08, 0, -0.04; //senseglove

    hmd_lhand_pose_.translation() += hmd_lhand_pose_.linear() * tracker_offset;
    hmd_rhand_pose_.translation() += hmd_rhand_pose_.linear() * tracker_offset;

    if ((hmd_check_pose_calibration_[0] == true) && (still_pose_cali_flag_ == false))
    {
        // hmd_still_cali_lhand_pos_ = hmd_lhand_pose_.translation() - hmd_chest_pose_.translation();
        // hmd_still_cali_rhand_pos_ = hmd_rhand_pose_.translation() - hmd_chest_pose_.translation();
        hmd_still_cali_lhand_pos_ = hmd_lhand_pose_.translation();
        hmd_still_cali_rhand_pos_ = hmd_rhand_pose_.translation();

        // hmd_still_cali_lhand_pos_ <<  -0.13517, 0.289845, -0.259223;
        // hmd_still_cali_rhand_pos_ <<   -0.0650479, -0.324795, -0.255538;

        std_msgs::String msg;
        std::stringstream still_cali_data;
        still_cali_data << "still_L : " << (hmd_still_cali_lhand_pos_(0)) << ", "
                        << (hmd_still_cali_lhand_pos_(1)) << ", "
                        << (hmd_still_cali_lhand_pos_(2)) << std::endl
                        << "still_R : " << (hmd_still_cali_rhand_pos_(0)) << ", "
                        << (hmd_still_cali_rhand_pos_(1)) << ", "
                        << (hmd_still_cali_rhand_pos_(2)) << std::endl;
        msg.data = still_cali_data.str();
        calibration_state_pub.publish(msg);
        calibration_state_gui_log_pub.publish(msg);

        cout << "hmd_still_cali_lhand_pos_: " << hmd_still_cali_lhand_pos_ << endl;
        cout << "hmd_still_cali_rhand_pos_: " << hmd_still_cali_rhand_pos_ << endl;

        calibration_log_file_ofstream_[0].open(calibration_folder_dir_ + "/still_pose_.txt");
        calibration_log_file_ofstream_[0] << hmd_still_cali_lhand_pos_ << endl;
        calibration_log_file_ofstream_[0] << hmd_still_cali_rhand_pos_ << endl;
        calibration_log_file_ofstream_[0].close();

        hmd_head_pose_init_ = hmd_head_pose_;
        hmd_lupperarm_pose_init_ = hmd_lupperarm_pose_;
        hmd_lhand_pose_init_ = hmd_lhand_pose_;
        hmd_rupperarm_pose_init_ = hmd_rupperarm_pose_;
        hmd_rhand_pose_init_ = hmd_rhand_pose_;
        hmd_pelv_pose_init_ = hmd_pelv_pose_;
        hmd_chest_pose_init_ = hmd_chest_pose_;

        // hmd_head_pose_init_ 	<<  -0.13517, 0.289845, -0.259223;
        // hmd_lupperarm_pose_init_<<  -0.13517, 0.289845, -0.259223;
        // hmd_lhand_pose_init_ 	<<  -0.13517, 0.289845, -0.259223;
        // hmd_rupperarm_pose_init_<<  -0.13517, 0.289845, -0.259223;
        // hmd_rhand_pose_init_ 	<<  -0.13517, 0.289845, -0.259223;
        // hmd_pelv_pose_init_ 	<<  -0.13517, 0.289845, -0.259223;
        // hmd_chest_pose_init_ 	<<  -0.13517, 0.289845, -0.259223;

        cout << "hmd_head_pose_init_: " << hmd_head_pose_init_.translation() << endl;
        cout << "hmd_lupperarm_pose_init_: " << hmd_lupperarm_pose_init_.translation() << endl;
        cout << "hmd_lhand_pose_init_: " << hmd_lhand_pose_init_.translation() << endl;
        cout << "hmd_rupperarm_pose_init_: " << hmd_rupperarm_pose_init_.translation() << endl;
        cout << "hmd_rhand_pose_init_: " << hmd_rhand_pose_init_.translation() << endl;
        cout << "hmd_pelv_pose_init_: " << hmd_pelv_pose_init_.translation() << endl;
        cout << "hmd_chest_pose_init_: " << hmd_chest_pose_init_.translation() << endl;

        calibration_log_file_ofstream_[3].open(calibration_folder_dir_ + "/hmd_pose_init_.txt");
        calibration_log_file_ofstream_[3] << hmd_head_pose_init_.translation() << "\n"
                                          << hmd_head_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3] << hmd_lupperarm_pose_init_.translation() << "\n"
                                          << hmd_lupperarm_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3] << hmd_lhand_pose_init_.translation() << "\n"
                                          << hmd_lhand_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3] << hmd_rupperarm_pose_init_.translation() << "\n"
                                          << hmd_rupperarm_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3] << hmd_rhand_pose_init_.translation() << "\n"
                                          << hmd_rhand_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3] << hmd_pelv_pose_init_.translation() << "\n"
                                          << hmd_pelv_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3] << hmd_chest_pose_init_.translation() << "\n"
                                          << hmd_chest_pose_init_.linear() << endl;
        calibration_log_file_ofstream_[3].close();
        still_pose_cali_flag_ = true;
    }

    if ((hmd_check_pose_calibration_[1] == true) && (t_pose_cali_flag_ == false))
    {
        // hmd_tpose_cali_lhand_pos_ = hmd_lhand_pose_.translation() - hmd_chest_pose_.translation();
        // hmd_tpose_cali_rhand_pos_ = hmd_rhand_pose_.translation() - hmd_chest_pose_.translation();
        hmd_tpose_cali_lhand_pos_ = hmd_lhand_pose_.translation();
        hmd_tpose_cali_rhand_pos_ = hmd_rhand_pose_.translation();
        // 20210209
        //  		hmd_tpose_cali_lhand_pos_: -0.284813
        //   0.803326
        //   0.356484
        //  hmd_tpose_cali_rhand_pos_: -0.146704
        //  -0.806712
        //   0.357469
        //  hmd_tpose_cali_lhand_pos_ <<  -0.284813, 0.803326, 0.356484;
        //  hmd_tpose_cali_rhand_pos_ <<  -0.146704, -0.806712, 0.357469;

        std_msgs::String msg;
        std::stringstream tpose_cali_data;
        tpose_cali_data << "T_L : " << (hmd_tpose_cali_lhand_pos_(0)) << ", "
                        << (hmd_tpose_cali_lhand_pos_(1)) << ", "
                        << (hmd_tpose_cali_lhand_pos_(2)) << std::endl
                        << "T_R : " << (hmd_tpose_cali_rhand_pos_(0)) << ", "
                        << (hmd_tpose_cali_rhand_pos_(1)) << ", "
                        << (hmd_tpose_cali_rhand_pos_(2));
        msg.data = tpose_cali_data.str();
        calibration_state_pub.publish(msg);
        calibration_state_gui_log_pub.publish(msg);

        cout << "hmd_tpose_cali_lhand_pos_: " << hmd_tpose_cali_lhand_pos_ << endl;
        cout << "hmd_tpose_cali_rhand_pos_: " << hmd_tpose_cali_rhand_pos_ << endl;

        calibration_log_file_ofstream_[1].open(calibration_folder_dir_ + "/t_pose_.txt");
        calibration_log_file_ofstream_[1] << hmd_tpose_cali_lhand_pos_ << endl;
        calibration_log_file_ofstream_[1] << hmd_tpose_cali_rhand_pos_ << endl;
        calibration_log_file_ofstream_[1].close();
        t_pose_cali_flag_ = true;
    }

    if ((hmd_check_pose_calibration_[2] == true) && (forward_pose_cali_flag_ == false))
    {
        // hmd_forward_cali_lhand_pos_ = hmd_lhand_pose_.translation() - hmd_chest_pose_.translation();
        // hmd_forward_cali_rhand_pos_ = hmd_rhand_pose_.translation() - hmd_chest_pose_.translation();
        hmd_forward_cali_lhand_pos_ = hmd_lhand_pose_.translation();
        hmd_forward_cali_rhand_pos_ = hmd_rhand_pose_.translation();

        // 20210209

        // 0.430654
        // 0.263043
        //  0.36878
        // hmd_forward_cali_rhand_pos_:  0.463099
        // -0.198173
        //  0.373117

        // hmd_forward_cali_lhand_pos_ <<  0.430654, 0.263043, 0.36878;
        // hmd_forward_cali_rhand_pos_ <<  0.463099, -0.198173, 0.373117;

        std_msgs::String msg;
        std::stringstream forward_cali_data;
        forward_cali_data << "Foward_L : " << hmd_forward_cali_lhand_pos_(0) << ", "
                          << (hmd_forward_cali_lhand_pos_(1)) << ", "
                          << (hmd_forward_cali_lhand_pos_(2)) << std::endl
                          << "Forward_R : " << (hmd_forward_cali_rhand_pos_(0)) << ", "
                          << (hmd_forward_cali_rhand_pos_(1)) << ", "
                          << (hmd_forward_cali_rhand_pos_(2));
        msg.data = forward_cali_data.str();
        calibration_state_pub.publish(msg);
        calibration_state_gui_log_pub.publish(msg);

        cout << "hmd_forward_cali_lhand_pos_: " << hmd_forward_cali_lhand_pos_ << endl;
        cout << "hmd_forward_cali_rhand_pos_: " << hmd_forward_cali_rhand_pos_ << endl;

        calibration_log_file_ofstream_[2].open(calibration_folder_dir_ + "/forward_pose_.txt");
        calibration_log_file_ofstream_[2] << hmd_forward_cali_lhand_pos_ << endl;
        calibration_log_file_ofstream_[2] << hmd_forward_cali_rhand_pos_ << endl;
        calibration_log_file_ofstream_[2].close();
        forward_pose_cali_flag_ = true;
    }

    if ((hmd_check_pose_calibration_[4] == true) && (read_cali_log_flag_ == false))
    {
        //////////read calibration log file/////////////////
        calibration_log_file_ifstream_[0].open(calibration_folder_dir_ + "/still_pose_.txt");
        calibration_log_file_ifstream_[1].open(calibration_folder_dir_ + "/t_pose_.txt");
        calibration_log_file_ifstream_[2].open(calibration_folder_dir_ + "/forward_pose_.txt");
        calibration_log_file_ifstream_[3].open(calibration_folder_dir_ + "/hmd_pose_init_.txt");

        if (calibration_log_file_ifstream_[0].is_open())
        {
            getTranslationDataFromText(calibration_log_file_ifstream_[0], hmd_still_cali_lhand_pos_);
            getTranslationDataFromText(calibration_log_file_ifstream_[0], hmd_still_cali_rhand_pos_);
            cout << "Still Pose is Uploaded: [(" << hmd_still_cali_lhand_pos_.transpose() << "), (" << hmd_still_cali_rhand_pos_.transpose() << ")]" << endl;
            calibration_log_file_ifstream_[0].close();
        }
        else
        {
            cout << "Still Pose Calibration File Is Not Opened!" << endl;
        }

        if (calibration_log_file_ifstream_[1].is_open())
        {
            getTranslationDataFromText(calibration_log_file_ifstream_[1], hmd_tpose_cali_lhand_pos_);
            getTranslationDataFromText(calibration_log_file_ifstream_[1], hmd_tpose_cali_rhand_pos_);
            cout << "T Pose is Uploaded: [(" << hmd_tpose_cali_lhand_pos_.transpose() << "), (" << hmd_tpose_cali_rhand_pos_.transpose() << ")]" << endl;
            calibration_log_file_ifstream_[1].close();
        }
        else
        {
            cout << "T Pose Calibration File Is Not Opened!" << endl;
        }

        if (calibration_log_file_ifstream_[2].is_open())
        {
            getTranslationDataFromText(calibration_log_file_ifstream_[2], hmd_forward_cali_lhand_pos_);
            getTranslationDataFromText(calibration_log_file_ifstream_[2], hmd_forward_cali_rhand_pos_);
            cout << "Forward Pose is Uploaded: [(" << hmd_forward_cali_lhand_pos_.transpose() << "), (" << hmd_forward_cali_rhand_pos_.transpose() << ")]" << endl;
            calibration_log_file_ifstream_[2].close();
        }
        else
        {
            cout << "Forward Pose Calibration File Is Not Opened!" << endl;
        }

        if (calibration_log_file_ifstream_[3].is_open())
        {
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_head_pose_init_);
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_lupperarm_pose_init_);
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_lhand_pose_init_);
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_rupperarm_pose_init_);
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_rhand_pose_init_);
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_pelv_pose_init_);
            getIsometry3dDataFromText(calibration_log_file_ifstream_[3], hmd_chest_pose_init_);
            calibration_log_file_ifstream_[3].close();

            cout << "hmd_head_pose_init_: " << hmd_head_pose_init_.translation().transpose() << endl;
            cout << "hmd_lupperarm_pose_init_: " << hmd_lupperarm_pose_init_.translation().transpose() << endl;
            cout << "hmd_lhand_pose_init_: " << hmd_lhand_pose_init_.translation().transpose() << endl;
            cout << "hmd_rupperarm_pose_init_: " << hmd_rupperarm_pose_init_.translation().transpose() << endl;
            cout << "hmd_rhand_pose_init_: " << hmd_rhand_pose_init_.translation().transpose() << endl;
            cout << "hmd_pelv_pose_init_: " << hmd_pelv_pose_init_.translation().transpose() << endl;
            cout << "hmd_chest_pose_init_: " << hmd_chest_pose_init_.translation().transpose() << endl;
            cout << "hmd_chest_pose_init_.linear(): " << hmd_chest_pose_init_.linear() << endl;
        }
        else
        {
            cout << "HMD Init Pose File Is NOT Opened!" << endl;
        }

        hmd_check_pose_calibration_[3] = false;

        read_cali_log_flag_ = true;
    }

    if ((hmd_check_pose_calibration_[3] == false) && (still_pose_cali_flag_ * t_pose_cali_flag_ * forward_pose_cali_flag_ == true))
    {
        // hmd_lshoulder_center_pos_(0) = (hmd_still_cali_lhand_pos_(0) + hmd_tpose_cali_lhand_pos_(0)) / 2;
        // // hmd_lshoulder_center_pos_(1) = (hmd_still_cali_lhand_pos_(1) + hmd_forward_cali_lhand_pos_(1))/2;
        // hmd_lshoulder_center_pos_(1) = hmd_still_cali_lhand_pos_(1);
        // hmd_lshoulder_center_pos_(2) = (hmd_tpose_cali_lhand_pos_(2) + hmd_forward_cali_lhand_pos_(2)) / 2;

        // hmd_rshoulder_center_pos_(0) = (hmd_still_cali_rhand_pos_(0) + hmd_tpose_cali_rhand_pos_(0)) / 2;
        // // hmd_rshoulder_center_pos_(1) = (hmd_still_cali_rhand_pos_(1) + hmd_forward_cali_rhand_pos_(1))/2;
        // hmd_rshoulder_center_pos_(1) = hmd_still_cali_rhand_pos_(1);
        // hmd_rshoulder_center_pos_(2) = (hmd_tpose_cali_rhand_pos_(2) + hmd_forward_cali_rhand_pos_(2)) / 2;

        //// Geometric Shoulder Calculation ///////////////////////////
        getCenterOfShoulderCali(hmd_still_cali_lhand_pos_, hmd_tpose_cali_lhand_pos_, hmd_forward_cali_lhand_pos_, hmd_lshoulder_center_pos_);
        getCenterOfShoulderCali(hmd_still_cali_rhand_pos_, hmd_tpose_cali_rhand_pos_, hmd_forward_cali_rhand_pos_, hmd_rshoulder_center_pos_);

        hmd_larm_max_l_ = 0;
        hmd_larm_max_l_ += (hmd_lshoulder_center_pos_ - hmd_still_cali_lhand_pos_).norm();
        hmd_larm_max_l_ += (hmd_lshoulder_center_pos_ - hmd_tpose_cali_lhand_pos_).norm();
        hmd_larm_max_l_ += (hmd_lshoulder_center_pos_ - hmd_forward_cali_lhand_pos_).norm();
        hmd_larm_max_l_ /= 3;
        // hmd_larm_max_l_ = 0.54;

        hmd_rarm_max_l_ = 0;
        hmd_rarm_max_l_ += (hmd_rshoulder_center_pos_ - hmd_still_cali_rhand_pos_).norm();
        hmd_rarm_max_l_ += (hmd_rshoulder_center_pos_ - hmd_tpose_cali_rhand_pos_).norm();
        hmd_rarm_max_l_ += (hmd_rshoulder_center_pos_ - hmd_forward_cali_rhand_pos_).norm();
        hmd_rarm_max_l_ /= 3;

        hmd_shoulder_width_ = (hmd_lshoulder_center_pos_ - hmd_rshoulder_center_pos_).norm();
        // hmd_shoulder_width_ = 0.521045;

        // hmd_rarm_max_l_ = 0.54;
        Eigen::Vector3d l_still_basis = hmd_still_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        Eigen::Vector3d l_tpose_basis = hmd_tpose_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        Eigen::Vector3d l_forward_basis = hmd_forward_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        Eigen::Vector3d l_angle_btw_bases;
        l_angle_btw_bases(0) = (l_still_basis.dot(l_tpose_basis)) / (l_still_basis.norm() * l_tpose_basis.norm());
        l_angle_btw_bases(0) = DyrosMath::minmax_cut(l_angle_btw_bases(0), -1.0, 1.0);
        l_angle_btw_bases(0) = acos(l_angle_btw_bases(0)) * RAD2DEG;

        l_angle_btw_bases(1) = (l_tpose_basis.dot(l_forward_basis)) / (l_tpose_basis.norm() * l_forward_basis.norm());
        l_angle_btw_bases(1) = DyrosMath::minmax_cut(l_angle_btw_bases(1), -1.0, 1.0);
        l_angle_btw_bases(1) = acos(l_angle_btw_bases(1)) * RAD2DEG;

        l_angle_btw_bases(2) = (l_forward_basis.dot(l_still_basis)) / (l_forward_basis.norm() * l_still_basis.norm());
        l_angle_btw_bases(2) = DyrosMath::minmax_cut(l_angle_btw_bases(2), -1.0, 1.0);
        l_angle_btw_bases(2) = acos(l_angle_btw_bases(2)) * RAD2DEG;

        Eigen::Vector3d r_still_basis = hmd_still_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        Eigen::Vector3d r_tpose_basis = hmd_tpose_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        Eigen::Vector3d r_forward_basis = hmd_forward_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        Eigen::Vector3d r_angle_btw_bases;
        r_angle_btw_bases(0) = (r_still_basis.dot(r_tpose_basis)) / (r_still_basis.norm() * r_tpose_basis.norm());
        r_angle_btw_bases(0) = DyrosMath::minmax_cut(r_angle_btw_bases(0), -1.0, 1.0);
        r_angle_btw_bases(0) = acos(r_angle_btw_bases(0)) * RAD2DEG;

        r_angle_btw_bases(1) = (r_tpose_basis.dot(r_forward_basis)) / (r_tpose_basis.norm() * r_forward_basis.norm());
        r_angle_btw_bases(1) = DyrosMath::minmax_cut(r_angle_btw_bases(1), -1.0, 1.0);
        r_angle_btw_bases(1) = acos(r_angle_btw_bases(1)) * RAD2DEG;

        r_angle_btw_bases(2) = (r_forward_basis.dot(r_still_basis)) / (r_forward_basis.norm() * r_still_basis.norm());
        r_angle_btw_bases(2) = DyrosMath::minmax_cut(r_angle_btw_bases(2), -1.0, 1.0);
        r_angle_btw_bases(2) = acos(r_angle_btw_bases(2)) * RAD2DEG;

        std_msgs::String msg;
        std::stringstream arm_length_data;
        arm_length_data << "Left Arm Length : " << hmd_larm_max_l_ << ", "
                        << "Left Arm Length : " << hmd_rarm_max_l_ << "\n"
                        << "hmd_lshoulder_center_pos_: " << hmd_lshoulder_center_pos_.transpose() << "\n"
                        << "hmd_rshoulder_center_pos_: " << hmd_rshoulder_center_pos_.transpose() << "\n"
                        << "l_still_basis: " << l_still_basis.transpose() << "\n"
                        << "l_tpose_basis: " << l_tpose_basis.transpose() << "\n"
                        << "l_forward_basis: " << l_forward_basis.transpose() << "\n"
                        << "r_still_basis: " << r_still_basis.transpose() << "\n"
                        << "r_tpose_basis: " << r_tpose_basis.transpose() << "\n"
                        << "r_forward_basis: " << r_forward_basis.transpose() << "\n"
                        <<  "l_angle_btw_bases: " << l_angle_btw_bases.transpose() << "\n"
                        << "r_angle_btw_bases: " << r_angle_btw_bases.transpose() << endl;

        msg.data = arm_length_data.str();
        calibration_state_pub.publish(msg);
        calibration_state_gui_log_pub.publish(msg);

        cout << "hmd_lshoulder_center_pos_: " << hmd_lshoulder_center_pos_.transpose() << endl;
        cout << "hmd_rshoulder_center_pos_: " << hmd_rshoulder_center_pos_.transpose() << endl;
        cout << cblue << "hmd_larm_max_l_: " << hmd_larm_max_l_ << endl;
        cout << "hmd_rarm_max_l_: " << hmd_rarm_max_l_ << creset << endl;
        cout << "hmd_shoulder_width_: " << hmd_shoulder_width_ << endl;
        hmd_check_pose_calibration_[3] = true;

        hmd_chest_2_lshoulder_center_pos_ = hmd_lshoulder_center_pos_ - hmd_chest_pose_init_.translation();
        hmd_chest_2_rshoulder_center_pos_ = hmd_rshoulder_center_pos_ - hmd_chest_pose_init_.translation();

        cout << "hmd_chest_2_lshoulder_center_pos_: " << hmd_chest_2_lshoulder_center_pos_.transpose() << endl;
        cout << "hmd_chest_2_rshoulder_center_pos_: " << hmd_chest_2_rshoulder_center_pos_.transpose() << endl;

        cout << "l_still_basis: " << l_still_basis.transpose() << ", norm: " << l_still_basis.norm() << endl;
        cout << "l_tpose_basis: " << l_tpose_basis.transpose() << ", norm: " << l_tpose_basis.norm() << endl;
        cout << "l_forward_basis: " << l_forward_basis.transpose() << ", norm: " << l_forward_basis.norm() << endl;

        cout << "r_still_basis: " << r_still_basis.transpose() << ", norm: " << r_still_basis.norm() << endl;
        cout << "r_tpose_basis: " << r_tpose_basis.transpose() << ", norm: " << r_tpose_basis.norm() << endl;
        cout << "r_forward_basis: " << r_forward_basis.transpose() << ", norm: " << r_forward_basis.norm() << endl;

        cout << cblue << "l_angles_btw_bases(degree); should be near 90degrees: " << l_angle_btw_bases.transpose() << endl;
        cout << "r_angles_btw_bases(degree); should be near 90degrees: " << r_angle_btw_bases.transpose() << creset << endl;

        hmd_lshoulder_pose_init_.translation() = hmd_chest_pose_.linear() * hmd_chest_pose_init_.linear().transpose() * hmd_chest_2_lshoulder_center_pos_ + hmd_chest_pose_init_.translation();
        hmd_lshoulder_pose_init_.linear() = hmd_chest_pose_init_.linear();
        hmd_rshoulder_pose_init_.translation() = hmd_chest_pose_.linear() * hmd_chest_pose_init_.linear().transpose() * hmd_chest_2_rshoulder_center_pos_ + hmd_chest_pose_init_.translation();
        hmd_rshoulder_pose_init_.linear() = hmd_chest_pose_init_.linear();
    }

    // Shoulder Data
    hmd_lshoulder_pose_.translation() = hmd_chest_pose_.linear() * hmd_chest_pose_init_.linear().transpose() * hmd_chest_2_lshoulder_center_pos_ + hmd_chest_pose_.translation();
    hmd_lshoulder_pose_.linear() = hmd_chest_pose_.linear();
    hmd_rshoulder_pose_.translation() = hmd_chest_pose_.linear() * hmd_chest_pose_init_.linear().transpose() * hmd_chest_2_rshoulder_center_pos_ + hmd_chest_pose_.translation();
    hmd_rshoulder_pose_.linear() = hmd_chest_pose_.linear();

    // HMD Velocity
    double tracker_hz = 130;

    hmd_head_vel_.segment(0, 3) = (hmd_head_pose_.translation() - hmd_head_pose_pre_.translation()) * tracker_hz;
    hmd_lshoulder_vel_.segment(0, 3) = (hmd_lshoulder_pose_.translation() - hmd_lshoulder_pose_pre_.translation()) * tracker_hz;
    hmd_lupperarm_vel_.segment(0, 3) = (hmd_lupperarm_pose_.translation() - hmd_lupperarm_pose_pre_.translation()) * tracker_hz;
    hmd_lhand_vel_.segment(0, 3) = (hmd_lhand_pose_.translation() - hmd_lhand_pose_pre_.translation()) * tracker_hz;
    hmd_rshoulder_vel_.segment(0, 3) = (hmd_rshoulder_pose_.translation() - hmd_rshoulder_pose_pre_.translation()) * tracker_hz;
    hmd_rupperarm_vel_.segment(0, 3) = (hmd_rupperarm_pose_.translation() - hmd_rupperarm_pose_pre_.translation()) * tracker_hz;
    hmd_rhand_vel_.segment(0, 3) = (hmd_rhand_pose_.translation() - hmd_rhand_pose_pre_.translation()) * tracker_hz;
    hmd_chest_vel_.segment(0, 3) = (hmd_chest_pose_.translation() - hmd_chest_pose_pre_.translation()) * tracker_hz;

    Eigen::AngleAxisd ang_temp_1(hmd_head_pose_.linear() * hmd_head_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_2(hmd_lshoulder_pose_.linear() * hmd_lshoulder_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_3(hmd_lupperarm_pose_.linear() * hmd_lupperarm_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_4(hmd_lhand_pose_.linear() * hmd_lhand_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_5(hmd_rshoulder_pose_.linear() * hmd_rshoulder_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_6(hmd_rupperarm_pose_.linear() * hmd_rupperarm_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_7(hmd_rhand_pose_.linear() * hmd_rhand_pose_pre_.linear().transpose());
    Eigen::AngleAxisd ang_temp_8(hmd_chest_pose_.linear() * hmd_chest_pose_pre_.linear().transpose());

    hmd_head_vel_.segment(3, 3) = ang_temp_1.axis() * ang_temp_1.angle() * tracker_hz;
    hmd_lshoulder_vel_.segment(3, 3) = ang_temp_2.axis() * ang_temp_2.angle() * tracker_hz;
    hmd_lupperarm_vel_.segment(3, 3) = ang_temp_3.axis() * ang_temp_3.angle() * tracker_hz;
    hmd_lhand_vel_.segment(3, 3) = ang_temp_4.axis() * ang_temp_4.angle() * tracker_hz;
    hmd_rshoulder_vel_.segment(3, 3) = ang_temp_5.axis() * ang_temp_5.angle() * tracker_hz;
    hmd_rupperarm_vel_.segment(3, 3) = ang_temp_6.axis() * ang_temp_6.angle() * tracker_hz;
    hmd_rhand_vel_.segment(3, 3) = ang_temp_7.axis() * ang_temp_7.angle() * tracker_hz;
    hmd_chest_vel_.segment(3, 3) = ang_temp_8.axis() * ang_temp_8.angle() * tracker_hz;

    double check_val = 0;

    check_val = hmd_lhand_vel_.segment(0, 3).norm();
    // abrupt motion check and stop

    double limit_val = 4.0;

    // if (upper_body_mode_>=5)
    if(false)
    {
        if (check_val > limit_val)
        {
            cout << cred << "WARNING: left hand linear velocity is over the 2.0m/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_lhand_vel_.segment(3, 3).norm();
        if ((check_val > limit_val * M_PI))
        {
            cout << cred <<"WARNING: left hand angular velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_rhand_vel_.segment(0, 3).norm();
        if ((check_val > limit_val))
        {
            cout << cred <<"WARNING: right hand linear velocity is over the 2.0m/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_rhand_vel_.segment(3, 3).norm();
        if ((check_val > limit_val * M_PI))
        {
            cout << cred <<"WARNING: right hand angular velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_lupperarm_vel_.segment(0, 3).norm();
        if ((check_val > limit_val))
        {
            cout << cred <<"WARNING: hmd_lupperarm_vel_ linear velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_lupperarm_vel_.segment(3, 3).norm();
        if ((check_val > limit_val * M_PI))
        {
            cout << cred <<"WARNING: hmd_lupperarm_vel_ angular velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_rupperarm_vel_.segment(0, 3).norm();
        if ((check_val > limit_val))
        {
            cout << cred <<"WARNING: hmd_rupperarm_vel_ linear velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_rupperarm_vel_.segment(3, 3).norm();
        if ((check_val > limit_val * M_PI))
        {
            cout << cred <<"WARNING: hmd_rupperarm_vel_ angular velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_head_vel_.segment(3, 3).norm();
        if ((check_val > limit_val * M_PI))
        {
            cout << cred <<"WARNING: Head angular velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }

        check_val = hmd_chest_vel_.segment(3, 3).norm();
        if ((check_val > limit_val * M_PI))
        {
            cout << cred <<"WARNING: Chest angular velocity is over the 360 degree/s limit" << check_val << creset << endl;
            avatarUpperbodyModeUpdate(3);
        }
    }
}
void AvatarController::getCenterOfShoulderCali(Eigen::Vector3d Still_pose_cali, Eigen::Vector3d T_pose_cali, Eigen::Vector3d Forward_pose_cali, Eigen::Vector3d &CenterOfShoulder_cali)
{
    Eigen::Matrix3d temp_mat;
    Eigen::Vector3d one3, normal_to_cali_plane, p1_p2, p2_p3, p3_p1, u12, v23, center_of_cali_plane1, center_of_cali_plane2;
    temp_mat << Still_pose_cali.transpose(), T_pose_cali.transpose(), Forward_pose_cali.transpose();
    one3(0) = 1;
    one3(1) = 1;
    one3(2) = 1;

    normal_to_cali_plane = (temp_mat.inverse()) * one3;

    p1_p2 = T_pose_cali - Still_pose_cali;
    p2_p3 = Forward_pose_cali - T_pose_cali;
    p3_p1 = Still_pose_cali - Forward_pose_cali;

    u12 = normal_to_cali_plane.cross(p1_p2);
    v23 = normal_to_cali_plane.cross(p2_p3);

    Eigen::MatrixXd temp_mat2, temp_vec2, ts;
    temp_mat2.resize(3, 2);
    temp_vec2.resize(3, 1);
    ts.resize(2, 1);

    temp_mat2 << u12, -v23;

    temp_vec2 = (Forward_pose_cali - Still_pose_cali) / 2;
    ts = (temp_mat2.transpose() * temp_mat2).inverse() * temp_mat2.transpose() * temp_vec2;

    center_of_cali_plane1 = (Still_pose_cali + T_pose_cali) / 2 + u12 * ts(0);
    center_of_cali_plane2 = (T_pose_cali + Forward_pose_cali) / 2 + v23 * ts(1);

    double r = (Still_pose_cali - center_of_cali_plane1).norm();

    double k1 = ((p1_p2.norm() * p1_p2.norm()) / 2 - r * r);
    k1 = sqrt(k1);
    double k2 = ((p2_p3.norm() * p2_p3.norm()) / 2 - r * r);
    k2 = sqrt(k2);
    double k3 = ((p3_p1.norm() * p3_p1.norm()) / 2 - r * r);
    k3 = sqrt(k3);
    double k_star = (k1 + k2 + k3) / 3;

    double k_threshold = 0.1;
    if ((abs(k1 - k2) > k_threshold) || (abs(k2 - k3) > k_threshold) || (abs(k1 - k3) > k_threshold))
    {
        cout << cred << "WARNING: Re-Calibration is REQUIRED!" << creset << endl;
    }
    CenterOfShoulder_cali = center_of_cali_plane1 - normal_to_cali_plane.normalized() * k_star;
}


void AvatarController::getTranslationDataFromText(std::ifstream &text_file, Eigen::Vector3d &trans)
{
    for (int i = 0; i < 3; i++)
    {
        string data;
        text_file >> data;
        trans(i) = atof(data.c_str());
    }
}
void AvatarController::getMatrix3dDataFromText(std::ifstream &text_file, Eigen::Matrix3d &mat)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            string data;
            text_file >> data;
            mat(i, j) = atof(data.c_str());
        }
    }
}

void AvatarController::getIsometry3dDataFromText(std::ifstream &text_file, Eigen::Isometry3d &isom)
{
    Vector3d trans;
    Matrix3d mat;
    getTranslationDataFromText(text_file, trans);
    getMatrix3dDataFromText(text_file, mat);
    isom.translation() = trans;
    isom.linear() = mat;
}

void AvatarController::rawMasterPoseProcessing()
{
    if (upperbody_mode_recieved_ == true)
    {
        upperbody_command_time_ = current_time_;
        upperbody_mode_q_init_ = motion_q_pre_;

        master_lhand_pose_ = lhand_transform_current_from_global_;
        master_rhand_pose_ = rhand_transform_current_from_global_;

        master_lhand_pose_pre_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_pre_ = rhand_transform_pre_desired_from_;
        master_lelbow_pose_pre_ = lupperarm_transform_pre_desired_from_;
        master_relbow_pose_pre_ = rupperarm_transform_pre_desired_from_;
        master_lshoulder_pose_pre_ = lacromion_transform_pre_desired_from_;
        master_rshoulder_pose_pre_ = racromion_transform_pre_desired_from_;
        master_head_pose_pre_ = head_transform_pre_desired_from_;
        master_upperbody_pose_pre_ = upperbody_transform_pre_desired_from_;

        master_lhand_pose_ppre_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_ppre_ = rhand_transform_pre_desired_from_;
        master_head_pose_ppre_ = head_transform_pre_desired_from_;
        master_lelbow_pose_ppre_ = lupperarm_transform_pre_desired_from_;
        master_relbow_pose_ppre_ = rupperarm_transform_pre_desired_from_;
        master_lshoulder_pose_ppre_ = lacromion_transform_pre_desired_from_;
        master_rshoulder_pose_ppre_ = racromion_transform_pre_desired_from_;
        master_upperbody_pose_ppre_ = upperbody_transform_pre_desired_from_;

        master_relative_lhand_pos_pre_ = lhand_transform_current_from_global_.translation() - rhand_transform_current_from_global_.translation();
        master_relative_rhand_pos_pre_ = rhand_transform_current_from_global_.translation() - lhand_transform_current_from_global_.translation();


        // save start variables for 3D mosue mode
        master_lhand_pose_start_ = lhand_transform_pre_desired_from_;
        master_rhand_pose_start_ = rhand_transform_pre_desired_from_;
        hmd_lhand_pose_start_ = hmd_lhand_pose_;
        hmd_rhand_pose_start_ = hmd_rhand_pose_;

        upperbody_mode_recieved_ = false;
        // hmd_shoulder_width_ = (hmd_lupperarm_pose_.translation() - hmd_rupperarm_pose_.translation()).norm();
    }

    // abruptMotionFilter();
    // hmdRawDataProcessing();
    if( upper_body_mode_ == 6  && upper_body_mode_ == 7)
    {
        handPositionRetargeting();
    }
    else if (upper_body_mode_ == 8)
    {
        /////Absolute hand position mapping //////
        Vector3d hand_offset;
        if(master_arm_mode_)
        {
            hand_offset << 0.0, 0, 0.0;
        }
        else
        {
            hand_offset << 0.0, 0.0, 0.15;
        }
        master_lhand_pose_raw_.translation() = hmd_lhand_pose_.translation() + hand_offset;
        master_rhand_pose_raw_.translation() = hmd_rhand_pose_.translation() + hand_offset;
        ///////////////////////////////////////////
    }
    else if (upper_body_mode_ == 9)
    {
        ///////Propotional hand position mapping////////////
        Vector3d hand_offset;
        if(master_arm_mode_)
        {
            hand_offset << 0.0, 0, 0.0;
        }
        else
        {
            hand_offset << 0.0, 0.0, 0.15;
        }
    
        master_lhand_pose_raw_.translation() = hand_pos_mapping_scale_raw_ * 1.3 * hmd_lhand_pose_.translation() + hand_offset;
        master_rhand_pose_raw_.translation() = hand_pos_mapping_scale_raw_ * 1.3 * hmd_rhand_pose_.translation() + hand_offset;

        //dg self collision test
        // master_lhand_pose_raw_.translation()(0) = 0.4 + 0.15*std::sin(current_time_*2*M_PI/4);
        // master_lhand_pose_raw_.translation()(1) = 0.05;
        // master_lhand_pose_raw_.translation()(2) = 0.2;
    }
    else if (upper_body_mode_ == 10)
    {
        ///////3D Mouse Mode////////////
        master_lhand_pose_raw_.translation() = master_lhand_pose_start_.translation() + 
                                                hand_pos_mapping_scale_raw_ * 1.3 * (hmd_lhand_pose_.translation() - hmd_lhand_pose_start_.translation());

        master_rhand_pose_raw_.translation() = master_rhand_pose_start_.translation() + 
                                                hand_pos_mapping_scale_raw_ * 1.3 * (hmd_rhand_pose_.translation() - hmd_rhand_pose_start_.translation());
        ////////////////////////////////////////////////////
    }

    orientationRetargeting();
    //////////////1025////////////////////////
    // master_lhand_pose_raw_ = master_lhand_pose_pre_;
    // master_rhand_pose_raw_ = master_rhand_pose_pre_;
    // master_lelbow_pose_raw_ = master_lelbow_pose_pre_;
    // master_relbow_pose_raw_ = master_relbow_pose_pre_;
    // master_lshoulder_pose_raw_= master_lshoulder_pose_pre_;
    // master_rshoulder_pose_raw_= master_rshoulder_pose_pre_;
    // master_head_pose_raw_= master_head_pose_pre_;
    // master_upperbody_pose_raw_= master_upperbody_pose_pre_;

    // master_lhand_pose_raw_.translation()(2) = DyrosMath::cubic(current_time_, upperbody_command_time_ +10, upperbody_command_time_+13, 0.2, 0.4, 0.0, 0.0);
    // master_lhand_pose_raw_.translation()(0) = 0.4;
    // master_lhand_pose_raw_.translation()(1) = 0.3;

    // master_rhand_pose_raw_.translation()(2) = DyrosMath::cubic(current_time_, upperbody_command_time_ +10, upperbody_command_time_+13, 0.2, 0.4, 0.0, 0.0);
    // master_rhand_pose_raw_.translation()(0) = 0.4;
    // master_rhand_pose_raw_.translation()(1) = -0.3;
    //////////////////////////////////////////

    double fc_filter = 10.0; // hz
    double spline_time = 3.0; // second
    if (current_time_ <= upperbody_command_time_ + spline_time)
    {
        for (int i = 0; i < 3; i++)
        {
            master_lhand_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lhand_transform_pre_desired_from_.translation()(i), 0, 0, master_lhand_pose_raw_.translation()(i), 0, 0)(0);
            master_rhand_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, rhand_transform_pre_desired_from_.translation()(i), 0, 0, master_rhand_pose_raw_.translation()(i), 0, 0)(0);

            master_lelbow_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lupperarm_transform_pre_desired_from_.translation()(i), 0, 0, master_lelbow_pose_raw_.translation()(i), 0, 0)(0);
            master_relbow_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, rupperarm_transform_pre_desired_from_.translation()(i), 0, 0, master_relbow_pose_raw_.translation()(i), 0, 0)(0);

            master_lshoulder_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lacromion_transform_pre_desired_from_.translation()(i), 0, 0, master_lshoulder_pose_raw_.translation()(i), 0, 0)(0);
            master_rshoulder_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, racromion_transform_pre_desired_from_.translation()(i), 0, 0, master_rshoulder_pose_raw_.translation()(i), 0, 0)(0);

            master_head_pose_raw_.translation()(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, head_transform_pre_desired_from_.translation()(i), 0, 0, master_head_pose_raw_.translation()(i), 0, 0)(0);

            master_relative_lhand_pos_raw_(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lhand_transform_pre_desired_from_.translation()(i) - rhand_transform_pre_desired_from_.translation()(i), 0, 0, master_relative_lhand_pos_raw_(i), 0, 0)(0);
            master_relative_rhand_pos_raw_(i) = DyrosMath::QuinticSpline(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, rhand_transform_pre_desired_from_.translation()(i) - lhand_transform_pre_desired_from_.translation()(i), 0, 0, master_relative_rhand_pos_raw_(i), 0, 0)(0);
        }

        master_lhand_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lhand_transform_pre_desired_from_.linear(), master_lhand_pose_raw_.linear());
        master_rhand_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, rhand_transform_pre_desired_from_.linear(), master_rhand_pose_raw_.linear());
        master_lelbow_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lupperarm_transform_pre_desired_from_.linear(), master_lelbow_pose_raw_.linear());
        master_relbow_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, rupperarm_transform_pre_desired_from_.linear(), master_relbow_pose_raw_.linear());
        master_lshoulder_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, lacromion_transform_pre_desired_from_.linear(), master_lshoulder_pose_raw_.linear());
        master_rshoulder_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, racromion_transform_pre_desired_from_.linear(), master_rshoulder_pose_raw_.linear());
        master_head_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, head_transform_pre_desired_from_.linear(), master_head_pose_raw_.linear());
        master_upperbody_pose_raw_.linear() = DyrosMath::rotationCubic(current_time_, upperbody_command_time_, upperbody_command_time_ + spline_time, upperbody_transform_pre_desired_from_.linear(), master_upperbody_pose_raw_.linear());
    }

    // master_lhand_pose_.translation() = DyrosMath::lpf<3>(master_lhand_pose_raw_.translation(), master_lhand_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_rhand_pose_.translation() = DyrosMath::lpf<3>(master_rhand_pose_raw_.translation(), master_rhand_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_lelbow_pose_.translation() = DyrosMath::lpf<3>(master_lelbow_pose_raw_.translation(), master_lelbow_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_relbow_pose_.translation() = DyrosMath::lpf<3>(master_relbow_pose_raw_.translation(), master_relbow_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_lshoulder_pose_.translation() = DyrosMath::lpf<3>(master_lshoulder_pose_raw_.translation(), master_lshoulder_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_rshoulder_pose_.translation() = DyrosMath::lpf<3>(master_rshoulder_pose_raw_.translation(), master_rshoulder_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_head_pose_.translation() = DyrosMath::lpf<3>(master_head_pose_raw_.translation(), master_head_pose_pre_.translation(), 1 / dt_, fc_filter);
    // master_upperbody_pose_.translation() = DyrosMath::lpf<3>(master_upperbody_pose_raw_.translation(), master_upperbody_pose_pre_.translation(), 1 / dt_, fc_filter);

    master_lhand_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_lhand_pose_raw_.translation(), master_lhand_pose_raw_pre_.translation(), master_lhand_pose_raw_ppre_.translation(), master_lhand_pose_pre_.translation(), master_lhand_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_rhand_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_rhand_pose_raw_.translation(), master_rhand_pose_raw_pre_.translation(), master_rhand_pose_raw_ppre_.translation(), master_rhand_pose_pre_.translation(), master_rhand_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_lelbow_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_lelbow_pose_raw_.translation(), master_lelbow_pose_raw_pre_.translation(), master_lelbow_pose_raw_ppre_.translation(), master_lelbow_pose_pre_.translation(), master_lelbow_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_relbow_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_relbow_pose_raw_.translation(), master_relbow_pose_raw_pre_.translation(), master_relbow_pose_raw_ppre_.translation(), master_relbow_pose_pre_.translation(), master_relbow_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_lshoulder_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_lshoulder_pose_raw_.translation(), master_lshoulder_pose_raw_pre_.translation(), master_lshoulder_pose_raw_ppre_.translation(), master_lshoulder_pose_pre_.translation(), master_lshoulder_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_rshoulder_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_rshoulder_pose_raw_.translation(), master_rshoulder_pose_raw_pre_.translation(), master_rshoulder_pose_raw_ppre_.translation(), master_rshoulder_pose_pre_.translation(), master_rshoulder_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_head_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_head_pose_raw_.translation(), master_head_pose_raw_pre_.translation(), master_head_pose_raw_ppre_.translation(), master_head_pose_pre_.translation(), master_head_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);
    master_upperbody_pose_.translation() = DyrosMath::secondOrderLowPassFilter<3>(master_upperbody_pose_raw_.translation(), master_upperbody_pose_raw_pre_.translation(), master_upperbody_pose_raw_ppre_.translation(), master_upperbody_pose_pre_.translation(), master_upperbody_pose_ppre_.translation(), fc_filter, 1, 1 / dt_);

    master_relative_lhand_pos_ = DyrosMath::lpf<3>(master_relative_lhand_pos_raw_, master_relative_lhand_pos_pre_, 1 / dt_, fc_filter);
    master_relative_rhand_pos_ = DyrosMath::lpf<3>(master_relative_rhand_pos_raw_, master_relative_rhand_pos_pre_, 1 / dt_, fc_filter);

    Eigen::AngleAxisd lhand_ang_diff(master_lhand_pose_raw_.linear() * master_lhand_pose_pre_.linear().transpose());
    Eigen::AngleAxisd rhand_ang_diff(master_rhand_pose_raw_.linear() * master_rhand_pose_pre_.linear().transpose());
    Eigen::AngleAxisd lelbow_ang_diff(master_lelbow_pose_raw_.linear() * master_lelbow_pose_pre_.linear().transpose());
    Eigen::AngleAxisd relbow_ang_diff(master_relbow_pose_raw_.linear() * master_relbow_pose_pre_.linear().transpose());
    Eigen::AngleAxisd lshoulder_ang_diff(master_lshoulder_pose_raw_.linear() * master_lshoulder_pose_pre_.linear().transpose());
    Eigen::AngleAxisd rshoulder_ang_diff(master_rshoulder_pose_raw_.linear() * master_rshoulder_pose_pre_.linear().transpose());
    Eigen::AngleAxisd head_ang_diff(master_head_pose_raw_.linear() * master_head_pose_pre_.linear().transpose());
    Eigen::AngleAxisd upperbody_ang_diff(master_upperbody_pose_raw_.linear() * master_upperbody_pose_pre_.linear().transpose());

    Eigen::Matrix3d lhand_diff_m, rhand_diff_m, lelbow_diff_m, relbow_diff_m, lshoulder_diff_m, rshoulder_diff_m, head_diff_m, upperbody_diff_m;
    double temp_ang_diff_filtered;
    // lhand_ang_diff.angle() = DyrosMath::lpf(lhand_ang_diff.angle(), 0, 2000, fc_filter);
    // lhand_diff_m = Eigen::AngleAxisd( temp_ang_diff_filtered, fc_filter), lhand_ang_diff.axis());
    lhand_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(lhand_ang_diff.angle(), 0, 1 / dt_, fc_filter), lhand_ang_diff.axis());
    rhand_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(rhand_ang_diff.angle(), 0, 1 / dt_, fc_filter), rhand_ang_diff.axis());
    lelbow_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(lelbow_ang_diff.angle(), 0, 1 / dt_, fc_filter), lelbow_ang_diff.axis());
    relbow_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(relbow_ang_diff.angle(), 0, 1 / dt_, fc_filter), relbow_ang_diff.axis());
    lshoulder_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(lshoulder_ang_diff.angle(), 0, 1 / dt_, fc_filter), lshoulder_ang_diff.axis());
    rshoulder_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(rshoulder_ang_diff.angle(), 0, 1 / dt_, fc_filter), rshoulder_ang_diff.axis());
    head_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(head_ang_diff.angle(), 0, 1 / dt_, fc_filter), head_ang_diff.axis());
    upperbody_diff_m = Eigen::AngleAxisd(DyrosMath::lpf(upperbody_ang_diff.angle(), 0, 1 / dt_, fc_filter), upperbody_ang_diff.axis());

    master_lhand_pose_.linear() = lhand_diff_m * master_lhand_pose_pre_.linear();
    master_rhand_pose_.linear() = rhand_diff_m * master_rhand_pose_pre_.linear();
    master_lelbow_pose_.linear() = lelbow_diff_m * master_lelbow_pose_pre_.linear();
    master_relbow_pose_.linear() = relbow_diff_m * master_relbow_pose_pre_.linear();
    master_lshoulder_pose_.linear() = lshoulder_diff_m * master_lshoulder_pose_pre_.linear();
    master_rshoulder_pose_.linear() = rshoulder_diff_m * master_rshoulder_pose_pre_.linear();
    master_head_pose_.linear() = head_diff_m * master_head_pose_pre_.linear();
    master_upperbody_pose_.linear() = upperbody_diff_m * master_upperbody_pose_pre_.linear();

    // for print
    master_lhand_rqy_ = DyrosMath::rot2Euler_tf(master_lhand_pose_.linear());
    master_rhand_rqy_ = DyrosMath::rot2Euler_tf(master_rhand_pose_.linear());

    master_lelbow_rqy_ = DyrosMath::rot2Euler_tf(master_lelbow_pose_.linear());
    master_relbow_rqy_ = DyrosMath::rot2Euler_tf(master_relbow_pose_.linear());

    master_lshoulder_rqy_ = DyrosMath::rot2Euler_tf(master_lshoulder_pose_.linear());
    master_rshoulder_rqy_ = DyrosMath::rot2Euler_tf(master_rshoulder_pose_.linear());

    master_head_rqy_ = DyrosMath::rot2Euler_tf(master_head_pose_.linear());

    // if( int(current_time_*10000)%1000 == 0)
    // {
    // 	cout<<"master_lelbow_rqy_: "<<master_lelbow_rqy_<<endl;
    // 	cout<<"master_relbow_rqy_: "<<master_relbow_rqy_<<endl;
    // }
    // master_lhand_pose_.linear() = lhand_transform_init_from_global_.linear();
    // master_rhand_pose_.linear() = rhand_transform_init_from_global_.linear();

    // double arm_len_max = 0.95;

    // if( master_lhand_pose_.translation().norm() > arm_len_max)
    // {
    // 	master_lhand_pose_.translation() = master_lhand_pose_.translation().normalized() * arm_len_max;
    // }

    // if( master_rhand_pose_.translation().norm() > arm_len_max)
    // {
    // 	master_rhand_pose_.translation() = master_rhand_pose_.translation().normalized() * arm_len_max;
    // }

    // for(int i = 0; i<3; i++)
    // {
    // 	master_lhand_vel_(i) = (master_lhand_pose_.translation()(i) - master_lhand_pose_pre_.translation()(i))/dt_;
    // 	master_rhand_vel_(i) = (master_rhand_pose_.translation()(i) - master_rhand_pose_pre_.translation()(i))/dt_;
    // }

    master_lhand_vel_.setZero();
    master_rhand_vel_.setZero();

    master_lelbow_vel_.setZero();
    master_relbow_vel_.setZero();

    master_lshoulder_vel_.setZero();
    master_rshoulder_vel_.setZero();

    master_head_vel_.setZero();
    master_upperbody_vel_.setZero();
}
void AvatarController::handPositionRetargeting()
{
    ///////////////////////////////////////////////HQP MOTION RETARGETING////////////////////////////////////////////
    if (first_loop_qp_retargeting_)
    {
        lhand_master_ref_stack_.setZero(3, 3);
        lhand_robot_ref_stack_.setZero(3, 3);
        rhand_master_ref_stack_.setZero(3, 3);
        rhand_robot_ref_stack_.setZero(3, 3);

        robot_still_pose_lhand_.setZero();
        robot_t_pose_lhand_.setZero();
        robot_forward_pose_lhand_.setZero();
        robot_still_pose_rhand_.setZero();
        robot_t_pose_rhand_.setZero();
        robot_forward_pose_rhand_.setZero();

        lhand_mapping_vector_.setZero();
        rhand_mapping_vector_.setZero();

        robot_still_pose_lhand_(2) += -(robot_arm_max_l_);
        robot_t_pose_lhand_(1) += (robot_arm_max_l_);
        robot_forward_pose_lhand_(0) += (robot_arm_max_l_);

        robot_still_pose_rhand_(2) += -(robot_arm_max_l_);
        robot_t_pose_rhand_(1) += -(robot_arm_max_l_);
        robot_forward_pose_rhand_(0) += (robot_arm_max_l_);

        lhand_master_ref_stack_.block(0, 0, 3, 1) = hmd_still_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        lhand_master_ref_stack_.block(0, 1, 3, 1) = hmd_tpose_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        lhand_master_ref_stack_.block(0, 2, 3, 1) = hmd_forward_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        // lhand_master_ref_stack_.block(0, 3, 3, 1) = hmd_tpose_cali_lhand_pos_ - hmd_lshoulder_center_pos_;
        // lhand_master_ref_stack_.block(0, 3, 3, 1) = hmd_rshoulder_center_pos_ - hmd_lshoulder_center_pos_;

        lhand_robot_ref_stack_.block(0, 0, 3, 1) = robot_still_pose_lhand_;
        lhand_robot_ref_stack_.block(0, 1, 3, 1) = robot_t_pose_lhand_;
        lhand_robot_ref_stack_.block(0, 2, 3, 1) = robot_forward_pose_lhand_;
        // lhand_robot_ref_stack_.block(0, 3, 3, 1) = robot_t_pose_lhand_;
        // lhand_robot_ref_stack_(1, 3) = -robot_shoulder_width_;

        rhand_master_ref_stack_.block(0, 0, 3, 1) = hmd_still_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        rhand_master_ref_stack_.block(0, 1, 3, 1) = hmd_tpose_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        rhand_master_ref_stack_.block(0, 2, 3, 1) = hmd_forward_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        // rhand_master_ref_stack_.block(0, 3, 3, 1) = hmd_tpose_cali_rhand_pos_ - hmd_rshoulder_center_pos_;
        // rhand_master_ref_stack_.block(0, 3, 3, 1) = hmd_lshoulder_center_pos_ - hmd_rshoulder_center_pos_;

        rhand_robot_ref_stack_.block(0, 0, 3, 1) = robot_still_pose_rhand_;
        rhand_robot_ref_stack_.block(0, 1, 3, 1) = robot_t_pose_rhand_;
        rhand_robot_ref_stack_.block(0, 2, 3, 1) = robot_forward_pose_rhand_;
        // rhand_robot_ref_stack_.block(0, 3, 3, 1) = robot_t_pose_rhand_;
        // rhand_robot_ref_stack_(1, 3) = robot_shoulder_width_;

        E1_.setZero(control_size_retargeting_[0], variable_size_retargeting_);
        E2_.setZero(control_size_retargeting_[1], variable_size_retargeting_);
        E3_.setZero(control_size_retargeting_[2], variable_size_retargeting_);
        H_retargeting_.setZero(variable_size_retargeting_, variable_size_retargeting_);
        g_retargeting_.setZero(variable_size_retargeting_);
        u1_.setZero(control_size_retargeting_[0]);
        u2_.setZero(control_size_retargeting_[1]);
        u3_.setZero(control_size_retargeting_[2]);

        ub_retargeting_.setZero(constraint_size1_retargeting_);
        lb_retargeting_.setZero(constraint_size1_retargeting_);

        E1_.block(0, 0, 3, 3) = lhand_master_ref_stack_;
        E2_.block(0, 3, 3, 3) = rhand_master_ref_stack_;
        E3_.block(0, 0, 3, 3) = lhand_robot_ref_stack_;
        E3_.block(0, 3, 3, 3) = -rhand_robot_ref_stack_;

        for (int i = 0; i < constraint_size1_retargeting_; i++)
        {
            ub_retargeting_(i) = w_dot_max_;
            lb_retargeting_(i) = w_dot_min_;
        }

        w1_retargeting_ = 1;
        w2_retargeting_ = 1;
        w3_retargeting_ = 1;
        human_shoulder_width_ = (hmd_rshoulder_center_pos_ - hmd_lshoulder_center_pos_).norm();

        Eigen::MatrixXd lhand_master_ref_stack_pinverse_ = lhand_master_ref_stack_.transpose() * (lhand_master_ref_stack_ * lhand_master_ref_stack_.transpose() + damped_puedoinverse_eps_ * Eigen::Matrix3d::Identity()).inverse();
        lhand_mapping_vector_pre_ = lhand_master_ref_stack_pinverse_ * hmd_lshoulder_pose_init_.linear() * hmd_lshoulder_pose_.linear().transpose() * (hmd_lhand_pose_.translation() - hmd_lshoulder_pose_.translation());

        Eigen::MatrixXd rhand_master_ref_stack_pinverse_ = rhand_master_ref_stack_.transpose() * (rhand_master_ref_stack_ * rhand_master_ref_stack_.transpose() + damped_puedoinverse_eps_ * Eigen::Matrix3d::Identity()).inverse();
        rhand_mapping_vector_pre_ = rhand_master_ref_stack_pinverse_ * hmd_rshoulder_pose_init_.linear() * hmd_rshoulder_pose_.linear().transpose() * (hmd_rhand_pose_.translation() - hmd_rshoulder_pose_.translation());

        h_pre_lhand_ = (hmd_lhand_pose_.translation() - hmd_lshoulder_pose_.translation());
        h_pre_rhand_ = (hmd_rhand_pose_.translation() - hmd_rshoulder_pose_.translation());

        for (int i = 0; i < 3; i++)
        {
            QP_motion_retargeting_[i].InitializeProblemSize(variable_size_retargeting_, constraint_size2_retargeting_[i]);

            A_retargeting_[i].setZero(constraint_size2_retargeting_[i], variable_size_retargeting_);
            ubA_retargeting_[i].setZero(constraint_size2_retargeting_[i]);
            lbA_retargeting_[i].setZero(constraint_size2_retargeting_[i]);

            A_retargeting_[i].block(0, 0, 3, 3) = lhand_master_ref_stack_;
            A_retargeting_[i].block(3, 3, 3, 3) = rhand_master_ref_stack_;

            for (int j = 0; j < 6; j++)
            {
                ubA_retargeting_[i](j) = human_vel_max_;
                lbA_retargeting_[i](j) = human_vel_min_;
            }

            qpres_retargeting_[i].setZero(variable_size_retargeting_);
        }

        first_loop_qp_retargeting_ = false;
    }
    else
    {
        double speed_reduce_rate = 20;

        ub_retargeting_(0) = min(speed_reduce_rate * (1.0 - lhand_mapping_vector_pre_(0)), w_dot_max_);
        ub_retargeting_(1) = min(speed_reduce_rate * (1.0 - lhand_mapping_vector_pre_(1)), w_dot_max_);
        ub_retargeting_(2) = min(speed_reduce_rate * (1.0 - lhand_mapping_vector_pre_(2)), w_dot_max_);
        // ub_retargeting_(3) = min(speed_reduce_rate * (1.0 - lhand_mapping_vector_pre_(3)), w_dot_max_);
        ub_retargeting_(3) = min(speed_reduce_rate * (1.0 - rhand_mapping_vector_pre_(0)), w_dot_max_);
        ub_retargeting_(4) = min(speed_reduce_rate * (1.0 - rhand_mapping_vector_pre_(1)), w_dot_max_);
        ub_retargeting_(5) = min(speed_reduce_rate * (1.0 - rhand_mapping_vector_pre_(2)), w_dot_max_);
        // ub_retargeting_(7) = min(speed_reduce_rate * (1.0 - rhand_mapping_vector_pre_(3)), w_dot_max_);

        lb_retargeting_(0) = max(speed_reduce_rate * (-1.1 - lhand_mapping_vector_pre_(0)), w_dot_min_);
        lb_retargeting_(1) = max(speed_reduce_rate * (-1.1 - lhand_mapping_vector_pre_(1)), w_dot_min_);
        lb_retargeting_(2) = max(speed_reduce_rate * (-1.1 - lhand_mapping_vector_pre_(2)), w_dot_min_);
        // lb_retargeting_(3) = max(speed_reduce_rate * (-1.0 - lhand_mapping_vector_pre_(3)), w_dot_min_);
        lb_retargeting_(3) = max(speed_reduce_rate * (-1.1 - rhand_mapping_vector_pre_(0)), w_dot_min_);
        lb_retargeting_(4) = max(speed_reduce_rate * (-1.1 - rhand_mapping_vector_pre_(1)), w_dot_min_);
        lb_retargeting_(5) = max(speed_reduce_rate * (-1.1 - rhand_mapping_vector_pre_(2)), w_dot_min_);
        // lb_retargeting_(7) = max(speed_reduce_rate * (-1.0 - rhand_mapping_vector_pre_(3)), w_dot_min_);

        h_pre_lhand_ = lhand_master_ref_stack_ * lhand_mapping_vector_pre_;
        h_pre_rhand_ = rhand_master_ref_stack_ * rhand_mapping_vector_pre_;
        r_pre_lhand_ = lhand_robot_ref_stack_ * lhand_mapping_vector_pre_;
        r_pre_rhand_ = rhand_robot_ref_stack_ * rhand_mapping_vector_pre_;
    }


    qpRetargeting_1(); // calc lhand_mapping_vector_, rhand_mapping_vector_ //1025



    hmd2robot_lhand_pos_mapping_ = lhand_robot_ref_stack_ * lhand_mapping_vector_;
    hmd2robot_rhand_pos_mapping_ = rhand_robot_ref_stack_ * rhand_mapping_vector_;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (hmd2robot_lhand_pos_mapping_.norm() > robot_arm_max_l_)
    {
        hmd2robot_lhand_pos_mapping_ = hmd2robot_lhand_pos_mapping_.normalized() * robot_arm_max_l_;
    }

    if (hmd2robot_rhand_pos_mapping_.norm() > robot_arm_max_l_)
    {
        hmd2robot_rhand_pos_mapping_ = hmd2robot_rhand_pos_mapping_.normalized() * robot_arm_max_l_;
    }

    if (hmd2robot_lhand_pos_mapping_.norm() < 0.1)
    {
        hmd2robot_lhand_pos_mapping_ = hmd2robot_lhand_pos_mapping_.normalized() * 0.1;
    }

    if (hmd2robot_rhand_pos_mapping_.norm() < 0.1)
    {
        hmd2robot_rhand_pos_mapping_ = hmd2robot_rhand_pos_mapping_.normalized() * 0.1;
    }


    Vector3d robot_init_hand_pos, robot_init_lshoulder_pos, robot_init_rshoulder_pos, delta_hmd2robot_lhand_pos_maping, delta_hmd2robot_rhand_pos_maping, delta_hmd2robot_lelbow_pos_maping, delta_hmd2robot_relbow_pos_maping;
    robot_init_hand_pos << 0, 0, -(robot_arm_max_l_);
    robot_init_lshoulder_pos << 0, 0.1491, 0.065;
    robot_init_rshoulder_pos << 0, -0.1491, 0.065;

    master_lhand_pose_raw_.translation() = larmbase_transform_pre_desired_from_.translation() + upperbody_transform_pre_desired_from_.linear() * (robot_init_lshoulder_pos + hmd2robot_lhand_pos_mapping_);
    master_rhand_pose_raw_.translation() = rarmbase_transform_pre_desired_from_.translation() + upperbody_transform_pre_desired_from_.linear() * (robot_init_rshoulder_pos + hmd2robot_rhand_pos_mapping_);
}
void AvatarController::orientationRetargeting()
{
     Matrix3d robot_lhand_ori_init, robot_rhand_ori_init, robot_lelbow_ori_init, robot_relbow_ori_init, robot_lshoulder_ori_init, robot_rshoulder_ori_init, robot_head_ori_init, robot_upperbody_ori_init;
    robot_lhand_ori_init = DyrosMath::rotateWithZ(-90 * DEG2RAD);
    robot_rhand_ori_init = DyrosMath::rotateWithZ(90 * DEG2RAD);
    // robot_lshoulder_ori_init = DyrosMath::rotateWithZ(-0.3);
    robot_lshoulder_ori_init.setIdentity();
    // robot_rshoulder_ori_init = DyrosMath::rotateWithZ(0.3);
    robot_rshoulder_ori_init.setIdentity();
    robot_head_ori_init.setIdentity();

    robot_head_ori_init = DyrosMath::rotateWithY(-10 * DEG2RAD);
    
    robot_upperbody_ori_init.setIdentity();

    // robot_lelbow_ori_init << 0, 0, -1, 1, 0, 0, 0, -1, 0;
    robot_lelbow_ori_init.setZero();
    robot_lelbow_ori_init(0, 2) = -1;
    robot_lelbow_ori_init(1, 0) = 1;
    robot_lelbow_ori_init(2, 1) = -1;
    robot_lelbow_ori_init = DyrosMath::rotateWithZ(-0 * DEG2RAD) * robot_lelbow_ori_init;

    robot_relbow_ori_init.setZero();
    robot_relbow_ori_init(0, 2) = -1;
    robot_relbow_ori_init(1, 0) = -1;
    robot_relbow_ori_init(2, 1) = 1;
    robot_relbow_ori_init = DyrosMath::rotateWithZ(0 * DEG2RAD) * robot_relbow_ori_init;


    master_upperbody_pose_raw_.translation().setZero();
    Eigen::Matrix3d chest_diff_m, shoulder_diff_m;
    Eigen::AngleAxisd chest_ang_diff(hmd_chest_pose_.linear() * hmd_chest_pose_init_.linear().transpose());
    chest_diff_m = Eigen::AngleAxisd(chest_ang_diff.angle() * 1.0, chest_ang_diff.axis());
    // master_upperbody_pose_raw_.linear() = chest_diff_m * robot_upperbody_ori_init;
    master_upperbody_pose_raw_.linear() = hmd_chest_pose_.linear()*hmd_chest_pose_init_.linear().transpose()*robot_upperbody_ori_init;

    master_lhand_pose_raw_.linear() = hmd_lhand_pose_.linear() * DyrosMath::rotateWithZ(M_PI / 2); // absolute orientation

    master_rhand_pose_raw_.linear() = hmd_rhand_pose_.linear() * DyrosMath::rotateWithZ(-M_PI / 2); // absolute orientation

    master_lelbow_pose_raw_.translation().setZero();
    master_lelbow_pose_raw_.linear() = hmd_lupperarm_pose_.linear() * hmd_lupperarm_pose_init_.linear().transpose() * robot_lelbow_ori_init;

    master_relbow_pose_raw_.translation().setZero();
    master_relbow_pose_raw_.linear() = hmd_rupperarm_pose_.linear() * hmd_rupperarm_pose_init_.linear().transpose() * robot_relbow_ori_init;

    master_lshoulder_pose_raw_.translation().setZero();
    master_lshoulder_pose_raw_.linear() = hmd_lshoulder_pose_.linear() * hmd_lshoulder_pose_init_.linear().transpose() * robot_lshoulder_ori_init;

    master_rshoulder_pose_raw_.translation().setZero();
    master_rshoulder_pose_raw_.linear() = hmd_rshoulder_pose_.linear() * hmd_rshoulder_pose_init_.linear().transpose() * robot_rshoulder_ori_init;

    Vector3d hmd_head_displacement = hmd_head_pose_.translation() - hmd_head_pose_init_.translation();
    hmd_head_displacement(0) = DyrosMath::minmax_cut(hmd_head_displacement(0), -0.10, +0.10);
    hmd_head_displacement(1) = DyrosMath::minmax_cut(hmd_head_displacement(1), -0.15, +0.15);

    master_head_pose_raw_.translation() = hmd_head_displacement;
    master_head_pose_raw_.translation()(0) += 0.10;
    master_head_pose_raw_.linear() = hmd_head_pose_.linear() * hmd_head_pose_init_.linear().transpose() * robot_head_ori_init;

    // master_head_pose_raw_.linear() = hmd_head_pose_.linear();

    shoulder_diff_m = Eigen::AngleAxisd(chest_ang_diff.angle() * 1.0, chest_ang_diff.axis());
    master_lshoulder_pose_raw_.linear() = shoulder_diff_m * robot_upperbody_ori_init;
    master_rshoulder_pose_raw_.linear() = shoulder_diff_m * robot_upperbody_ori_init;
}

void AvatarController::qpRetargeting_1()
{
    h_d_lhand_ = hmd_chest_pose_init_.linear() * hmd_chest_pose_.linear().transpose() * (hmd_lhand_pose_.translation() - hmd_lshoulder_pose_.translation());
    h_d_rhand_ = hmd_chest_pose_init_.linear() * hmd_chest_pose_.linear().transpose() * (hmd_rhand_pose_.translation() - hmd_rshoulder_pose_.translation());

    u1_ = control_gain_retargeting_ * (h_d_lhand_ - h_pre_lhand_);
    u2_ = control_gain_retargeting_ * (h_d_rhand_ - h_pre_rhand_);

    H_retargeting_ = w1_retargeting_ * E1_.transpose() * E1_ + w2_retargeting_ * E2_.transpose() * E2_ + Eigen::MatrixXd::Identity(6, 6) * damped_puedoinverse_eps_;
    g_retargeting_ = -w1_retargeting_ * E1_.transpose() * u1_ - w2_retargeting_ * E2_.transpose() * u2_;

    QP_motion_retargeting_[0].EnableEqualityCondition(equality_condition_eps_);
    QP_motion_retargeting_[0].UpdateMinProblem(H_retargeting_, g_retargeting_);
    QP_motion_retargeting_[0].UpdateSubjectToAx(A_retargeting_[0], lbA_retargeting_[0], ubA_retargeting_[0]);
    QP_motion_retargeting_[0].UpdateSubjectToX(lb_retargeting_, ub_retargeting_);

    // if (int(current_time_ * 10000) % 1000 == 0)
    // {
    //     cout << "lb_retargeting_: " << lb_retargeting_.transpose() << endl;
    //     cout << "ub_retargeting_: " << ub_retargeting_.transpose() << endl;
    // }

    if (QP_motion_retargeting_[0].SolveQPoases(200, qpres_retargeting_[0]))
    {
        lhand_mapping_vector_dot_ = qpres_retargeting_[0].segment(0, 3);
        rhand_mapping_vector_dot_ = qpres_retargeting_[0].segment(3, 3);

        lhand_mapping_vector_ = lhand_mapping_vector_pre_ + lhand_mapping_vector_dot_ * dt_;
        rhand_mapping_vector_ = rhand_mapping_vector_pre_ + rhand_mapping_vector_dot_ * dt_;
    }
    else
    {
        QP_motion_retargeting_[0].InitializeProblemSize(variable_size_retargeting_, constraint_size2_retargeting_[0]);
        lhand_mapping_vector_ = lhand_mapping_vector_pre_;
        rhand_mapping_vector_ = rhand_mapping_vector_pre_;
        if (int(current_time_ * 2000) % 1000 == 0)
            cout << "QP motion retargetng is not solved!! (beta == 0)" << endl;
    }
}

void AvatarController::loadCollisionThreshold(std::string folder_path)
{
    std::string thr_path("collision_threshold.txt");

    thr_path = folder_path + thr_path;
    col_thr_file_.open(thr_path, ios::in);

    int index = 0;
    float temp;
    threshold_joint_torque_collision_.setZero();

    if (!col_thr_file_.is_open())
    {
        std::cout << "Can not find the Collision Threshold file" << std::endl;
    }
    else
    {
        while (!col_thr_file_.eof())
        {
            col_thr_file_ >> temp;

            if (temp == temp)
            {
                if (index < 12)
                {
                    threshold_joint_torque_collision_(index) = temp;
                    index++;
                    cout << "threshold: " << index << ", " << temp << endl;
                }
                else
                {
                    cout << "Collision Threshold file has more than 12 values" << endl;
                }
            }
            else
            {
                cout << "WARNING: collision_threshold has NaN value! (" << temp << ") at" + thr_path << endl;
            }
        }
    }

    if (index == 12)
    {
        cout << "Collision Threshold: [" << threshold_joint_torque_collision_.segment(0, 12).transpose() << "]" << endl;
    }
    col_thr_file_.close();

    maximum_collision_free_torque_ = threshold_joint_torque_collision_;
}

void AvatarController::savePreData()
{
    pre_time_ = current_time_;
    pre_q_ = rd_.q_;
    pre_desired_q_ = desired_q_;
    pre_desired_q_dot_ = desired_q_dot_;
    motion_q_pre_ = motion_q_;
    motion_q_dot_pre_ = motion_q_dot_;


    master_lhand_pose_raw_ppre_ = master_lhand_pose_raw_pre_;
    master_lhand_pose_raw_ppre_ = master_lhand_pose_raw_pre_;
    master_rhand_pose_raw_ppre_ = master_rhand_pose_raw_pre_;
    master_head_pose_raw_ppre_ = master_head_pose_raw_pre_;
    master_lelbow_pose_raw_ppre_ = master_lelbow_pose_raw_pre_;
    master_relbow_pose_raw_ppre_ = master_relbow_pose_raw_pre_;
    master_lshoulder_pose_raw_ppre_ = master_lshoulder_pose_raw_pre_;
    master_rshoulder_pose_raw_ppre_ = master_rshoulder_pose_raw_pre_;
    master_upperbody_pose_raw_ppre_ = master_upperbody_pose_raw_pre_;

    master_lhand_pose_raw_pre_ = master_lhand_pose_raw_;
    master_rhand_pose_raw_pre_ = master_rhand_pose_raw_;
    master_head_pose_raw_pre_ = master_head_pose_raw_;
    master_lelbow_pose_raw_pre_ = master_lelbow_pose_raw_;
    master_relbow_pose_raw_pre_ = master_relbow_pose_raw_;
    master_lshoulder_pose_raw_pre_ = master_lshoulder_pose_raw_;
    master_rshoulder_pose_raw_pre_ = master_rshoulder_pose_raw_;
    master_upperbody_pose_raw_pre_ = master_upperbody_pose_raw_;

    master_lhand_pose_ppre_ = master_lhand_pose_pre_;
    master_rhand_pose_ppre_ = master_rhand_pose_pre_;
    master_head_pose_ppre_ = master_head_pose_pre_;
    master_lelbow_pose_ppre_ = master_lelbow_pose_pre_;
    master_relbow_pose_ppre_ = master_relbow_pose_pre_;
    master_lshoulder_pose_ppre_ = master_lshoulder_pose_pre_;
    master_rshoulder_pose_ppre_ = master_rshoulder_pose_pre_;
    master_upperbody_pose_ppre_ = master_upperbody_pose_pre_;

    master_lhand_pose_pre_ = master_lhand_pose_;
    master_rhand_pose_pre_ = master_rhand_pose_;
    master_lelbow_pose_pre_ = master_lelbow_pose_;
    master_relbow_pose_pre_ = master_relbow_pose_;
    master_lshoulder_pose_pre_ = master_lshoulder_pose_;
    master_rshoulder_pose_pre_ = master_rshoulder_pose_;
    master_head_pose_pre_ = master_head_pose_;
    master_upperbody_pose_pre_ = master_upperbody_pose_;

    master_relative_lhand_pos_pre_ = master_relative_lhand_pos_;
    master_relative_rhand_pos_pre_ = master_relative_rhand_pos_;

    hmd_tracker_status_pre_ = hmd_tracker_status_;

    hmd_head_pose_pre_ = hmd_head_pose_;
    hmd_lshoulder_pose_pre_ = hmd_lshoulder_pose_;
    hmd_lupperarm_pose_pre_ = hmd_lupperarm_pose_;
    hmd_lhand_pose_pre_ = hmd_lhand_pose_;
    hmd_rshoulder_pose_pre_ = hmd_rshoulder_pose_;
    hmd_rupperarm_pose_pre_ = hmd_rupperarm_pose_;
    hmd_rhand_pose_pre_ = hmd_rhand_pose_;
    hmd_chest_pose_pre_ = hmd_chest_pose_;
    hmd_pelv_pose_pre_ = hmd_pelv_pose_;

    lhand_mapping_vector_pre_ = lhand_mapping_vector_;
    rhand_mapping_vector_pre_ = rhand_mapping_vector_;

    avatar_op_pedal_pre_ = avatar_op_pedal_;
}

void AvatarController::UpperbodyModeCallback(const std_msgs::Int8 &msg)
{
    // avatarUpperbodyModeUpdate(msg.data);
    upper_body_mode_raw_ = msg.data;
    upperbody_mode_recieved_ = true;
}

void AvatarController::ArmJointGainCallback(const std_msgs::Float32MultiArray &msg)
{
    // left arm kp
    kp_joint_(15) = msg.data[0];
    kp_joint_(16) = msg.data[1];
    kp_joint_(17) = msg.data[2];
    kp_joint_(18) = msg.data[3];
    kp_joint_(19) = msg.data[4];
    kp_joint_(20) = msg.data[5];
    kp_joint_(21) = msg.data[6];
    kp_joint_(22) = msg.data[7];
    // right arm kp
    kp_joint_(25) = msg.data[0];
    kp_joint_(26) = msg.data[1];
    kp_joint_(27) = msg.data[2];
    kp_joint_(28) = msg.data[3];
    kp_joint_(29) = msg.data[4];
    kp_joint_(30) = msg.data[5];
    kp_joint_(31) = msg.data[6];
    kp_joint_(32) = msg.data[7];

    // left arm kd
    kv_joint_(15) = msg.data[8];
    kv_joint_(16) = msg.data[9];
    kv_joint_(17) = msg.data[10];
    kv_joint_(18) = msg.data[11];
    kv_joint_(19) = msg.data[12];
    kv_joint_(20) = msg.data[13];
    kv_joint_(21) = msg.data[14];
    kv_joint_(22) = msg.data[15];
    // right arm kd
    kv_joint_(25) = msg.data[8];
    kv_joint_(26) = msg.data[9];
    kv_joint_(27) = msg.data[10];
    kv_joint_(28) = msg.data[11];
    kv_joint_(29) = msg.data[12];
    kv_joint_(30) = msg.data[13];
    kv_joint_(31) = msg.data[14];
    kv_joint_(32) = msg.data[15];
}

void AvatarController::WaistJointGainCallback(const std_msgs::Float32MultiArray &msg)
{
    kp_joint_(12) = msg.data[0];
    kp_joint_(13) = msg.data[1];
    kp_joint_(14) = msg.data[2];

    kv_joint_(12) = msg.data[3];
    kv_joint_(13) = msg.data[4];
    kv_joint_(14) = msg.data[5];
}

void AvatarController::PoseCalibrationCallback(const std_msgs::Int8 &msg)
{
    if (msg.data == 1) // still pose
    {
        hmd_check_pose_calibration_[0] = true;
        cout << "Still Pose Calibration is On." << endl;
    }
    else if (msg.data == 2) // T pose
    {
        hmd_check_pose_calibration_[1] = true;
        cout << "T Pose Calibration is On." << endl;
    }
    else if (msg.data == 3) // forward stretch
    {
        hmd_check_pose_calibration_[2] = true;
        cout << "Forward Stretch Pose Calibration is On." << endl;
    }
    else if (msg.data == 4) // reset callibration
    {
        for (int i = 0; i < 5; i++)
        {
            hmd_check_pose_calibration_[i] = false;
        }
        still_pose_cali_flag_ = false;
        t_pose_cali_flag_ = false;
        forward_pose_cali_flag_ = false;
        read_cali_log_flag_ = false;
        hmd_check_pose_calibration_[3] = false;
        cout << "Pose Calibration is Reset." << endl;

        std_msgs::String msg;
        std::stringstream reset;
        reset << "RESET POSE CALIBRATION";
        msg.data = reset.str();
        calibration_state_pub.publish(msg);
        calibration_state_gui_log_pub.publish(msg);
    }
    else if (msg.data == 5)
    {
        hmd_check_pose_calibration_[0] = true;
        hmd_check_pose_calibration_[1] = true;
        hmd_check_pose_calibration_[2] = true;
        hmd_check_pose_calibration_[4] = true;

        still_pose_cali_flag_ = true;
        t_pose_cali_flag_ = true;
        forward_pose_cali_flag_ = true;
        cout << "Reading Calibration Log File..." << endl;

        std_msgs::String msg;
        std::stringstream log_load;
        log_load << "Calibration Pose Data is Loaded";
        msg.data = log_load.str();
        calibration_state_pub.publish(msg);
        calibration_state_gui_log_pub.publish(msg);
    }

    cout << "Calibration Status: [" << hmd_check_pose_calibration_[0] << ", " << hmd_check_pose_calibration_[1] << ", " << hmd_check_pose_calibration_[2] << "]" << endl;
}

void AvatarController::TrackerStatusCallback(const std_msgs::Bool &msg)
{
    hmd_tracker_status_raw_ = msg.data;
}

void AvatarController::TrackerPoseCallback(const geometry_msgs::PoseArray &msg)
{
    // msg.poses[0];

    if(master_arm_mode_ == true)
    {
        tf::poseMsgToEigen(msg.poses[0],hmd_pelv_pose_raw_);
    
        hmd_pelv_pose_raw_.linear() = hmd_pelv_pose_raw_.linear() * DyrosMath::rotateWithZ(M_PI); // tracker is behind the chair

        tf::poseMsgToEigen(msg.poses[1],hmd_chest_pose_raw_);
        tf::poseMsgToEigen(msg.poses[2],hmd_lupperarm_pose_raw_);
        tf::poseMsgToEigen(msg.poses[3],hmd_rupperarm_pose_raw_);
        tf::poseMsgToEigen(msg.poses[4],hmd_head_pose_raw_);
    }
    else
    {
        tf::poseMsgToEigen(msg.poses[0],hmd_pelv_pose_raw_);
    
        hmd_pelv_pose_raw_.linear() = hmd_pelv_pose_raw_.linear() * DyrosMath::rotateWithZ(M_PI); // tracker is behind the chair

        tf::poseMsgToEigen(msg.poses[1],hmd_chest_pose_raw_);
        tf::poseMsgToEigen(msg.poses[2],hmd_lupperarm_pose_raw_);
        tf::poseMsgToEigen(msg.poses[3],hmd_lhand_pose_raw_);
        tf::poseMsgToEigen(msg.poses[4],hmd_rupperarm_pose_raw_);
        tf::poseMsgToEigen(msg.poses[5],hmd_rhand_pose_raw_);
        tf::poseMsgToEigen(msg.poses[6],hmd_head_pose_raw_);       
    }
}

void AvatarController::MasterPoseCallback(const geometry_msgs::PoseArray &msg)
{
    if(master_arm_mode_ == true)
    {
        tf::poseMsgToEigen(msg.poses[0], hmd_lhand_pose_raw_);
        tf::poseMsgToEigen(msg.poses[1], hmd_rhand_pose_raw_);
    }
}
void AvatarController::HandPosMappingScaleCallback(const std_msgs::Float32 &msg)
{
    hand_pos_mapping_scale_raw_ = msg.data;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////// RL Walking Controller //////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

void AvatarController::loadNetwork()
{
    state_.setZero();
    rl_action_.setZero();


    string cur_path = "/home/cha/catkin_ws/src/tocabi_cc/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
    }

    base_path = loadPathFromConfig(cur_path + "weight_directory.txt");


    std::ifstream file[14];
    // file[0].open(cur_path+"weight/a2c_network_actor_mlp_0_weight.txt", std::ios::in);
    // file[1].open(cur_path+"weight/a2c_network_actor_mlp_0_bias.txt", std::ios::in);
    // file[2].open(cur_path+"weight/a2c_network_actor_mlp_2_weight.txt", std::ios::in);
    // file[3].open(cur_path+"weight/a2c_network_actor_mlp_2_bias.txt", std::ios::in);
    // file[4].open(cur_path+"weight/a2c_network_mu_weight.txt", std::ios::in);
    // file[5].open(cur_path+"weight/a2c_network_mu_bias.txt", std::ios::in);
    // file[6].open(cur_path+"weight/obs_mean_fixed.txt", std::ios::in);
    // file[7].open(cur_path+"weight/obs_variance_fixed.txt", std::ios::in);
    // file[8].open(cur_path+"weight/a2c_network_critic_mlp_0_weight.txt", std::ios::in);
    // file[9].open(cur_path+"weight/a2c_network_critic_mlp_0_bias.txt", std::ios::in);
    // file[10].open(cur_path+"weight/a2c_network_critic_mlp_2_weight.txt", std::ios::in);
    // file[11].open(cur_path+"weight/a2c_network_critic_mlp_2_bias.txt", std::ios::in);
    // file[12].open(cur_path+"weight/a2c_network_value_weight.txt", std::ios::in);
    // file[13].open(cur_path+"weight/a2c_network_value_bias.txt", std::ios::in);
    file[0].open(base_path + "policy/0_weight.txt", std::ios::in);
    file[1].open(base_path + "policy/0_bias.txt", std::ios::in);
    file[2].open(base_path + "policy/2_weight.txt", std::ios::in);
    file[3].open(base_path + "policy/2_bias.txt", std::ios::in);
    file[4].open(base_path + "policy/4_weight.txt", std::ios::in);
    file[5].open(base_path + "policy/4_bias.txt", std::ios::in);
    file[6].open(base_path + "normalizer/running_mean.txt", std::ios::in);
    file[7].open(base_path + "normalizer/running_var.txt", std::ios::in);
    file[8].open(base_path + "critic/0_weight.txt", std::ios::in);
    file[9].open(base_path + "critic/0_bias.txt", std::ios::in);
    file[10].open(base_path + "critic/2_weight.txt", std::ios::in);
    file[11].open(base_path + "critic/2_bias.txt", std::ios::in);
    file[12].open(base_path + "critic/4_weight.txt", std::ios::in);
    file[13].open(base_path + "critic/4_bias.txt", std::ios::in);

    if(!file[0].is_open())
    {
        std::cout<<"Can not find the weight file"<<std::endl;
    }

    float temp;
    int row = 0;
    int col = 0;

    while(!file[0].eof() && row != policy_net_w0_.rows())
    {
        file[0] >> temp;
        if(temp != '\n')
        {
            policy_net_w0_(row, col) = temp;
            col ++;
            if (col == policy_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[1].eof() && row != policy_net_b0_.rows())
    {
        file[1] >> temp;
        if(temp != '\n')
        {
            policy_net_b0_(row, col) = temp;
            col ++;
            if (col == policy_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[2].eof() && row != policy_net_w2_.rows())
    {
        file[2] >> temp;
        if(temp != '\n')
        {
            policy_net_w2_(row, col) = temp;
            col ++;
            if (col == policy_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[3].eof() && row != policy_net_b2_.rows())
    {
        file[3] >> temp;
        if(temp != '\n')
        {
            policy_net_b2_(row, col) = temp;
            col ++;
            if (col == policy_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[4].eof() && row != action_net_w_.rows())
    {
        file[4] >> temp;
        if(temp != '\n')
        {
            action_net_w_(row, col) = temp;
            col ++;
            if (col == action_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[5].eof() && row != action_net_b_.rows())
    {
        file[5] >> temp;
        if(temp != '\n')
        {
            action_net_b_(row, col) = temp;
            col ++;
            if (col == action_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[6].eof() && row != state_mean_.rows())
    {
        file[6] >> temp;
        if(temp != '\n')
        {
            state_mean_(row, col) = temp;
            col ++;
            if (col == state_mean_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[7].eof() && row != state_var_.rows())
    {
        file[7] >> temp;
        if(temp != '\n')
        {
            state_var_(row, col) = temp + 1.e-4;
            col ++;
            if (col == state_var_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[8].eof() && row != value_net_w0_.rows())
    {
        file[8] >> temp;
        if(temp != '\n')
        {
            value_net_w0_(row, col) = temp;
            col ++;
            if (col == value_net_w0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[9].eof() && row != value_net_b0_.rows())
    {
        file[9] >> temp;
        if(temp != '\n')
        {
            value_net_b0_(row, col) = temp;
            col ++;
            if (col == value_net_b0_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[10].eof() && row != value_net_w2_.rows())
    {
        file[10] >> temp;
        if(temp != '\n')
        {
            value_net_w2_(row, col) = temp;
            col ++;
            if (col == value_net_w2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[11].eof() && row != value_net_b2_.rows())
    {
        file[11] >> temp;
        if(temp != '\n')
        {
            value_net_b2_(row, col) = temp;
            col ++;
            if (col == value_net_b2_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[12].eof() && row != value_net_w_.rows())
    {
        file[12] >> temp;
        if(temp != '\n')
        {
            value_net_w_(row, col) = temp;
            col ++;
            if (col == value_net_w_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    row = 0;
    col = 0;
    while(!file[13].eof() && row != value_net_b_.rows())
    {
        file[13] >> temp;
        if(temp != '\n')
        {
            value_net_b_(row, col) = temp;
            col ++;
            if (col == value_net_b_.cols())
            {
                col = 0;
                row ++;
            }
        }
    }
    if (use_encoder_) loadEncoderNetwork();
}

void AvatarController::initVariable()
{    
    // Load the path from the configuration file

    policy_net_w0_.resize(num_hidden1, policy_input_dim_);
    policy_net_b0_.resize(num_hidden1, 1);
    policy_net_w2_.resize(num_hidden2, num_hidden1);
    policy_net_b2_.resize(num_hidden2, 1);
    action_net_w_.resize(num_action, num_hidden2);
    action_net_b_.resize(num_action, 1);
    hidden_layer1_.resize(num_hidden1, 1);
    hidden_layer2_.resize(num_hidden2, 1);
    rl_action_.resize(num_action, 1);

    value_net_w0_.resize(num_hidden1, policy_input_dim_);
    value_net_b0_.resize(num_hidden1, 1);
    value_net_w2_.resize(num_hidden2, num_hidden1);
    value_net_b2_.resize(num_hidden2, 1);
    value_net_w_.resize(1, num_hidden2);
    value_net_b_.resize(1, 1);
    value_hidden_layer1_.resize(num_hidden1, 1);
    value_hidden_layer2_.resize(num_hidden2, 1);
    
    // state_cur_.resize(num_cur_state, 1);
    state_cur_ = MatrixXd::Zero(num_cur_state, 1);
    state_.resize(num_state, 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    state_mean_.resize(num_cur_state, 1);
    state_var_.resize(num_cur_state, 1);

    // Encoder

    state_history_.resize(num_cur_state, history_len_ * history_skip_);

    encoder_input_.resize(num_cur_state, history_len_);
    encoder_conv1_w_.resize(encoder_conv1_output_channel_, encoder_conv1_input_channel_*encoder_conv1_kernel_);
    encoder_conv1_b_.resize(encoder_conv1_output_channel_, 1);

    encoder_conv2_w_.resize(encoder_conv2_output_channel_, encoder_conv2_input_channel_*encoder_conv2_kernel_);
    encoder_conv2_b_.resize(encoder_conv2_output_channel_, 1);

    encoder_fc_w_.resize(encoder_dim_, encoder_conv2_output_channel_*encoder_conv2_output_size_);
    encoder_fc_b_.resize(encoder_dim_, 1);

    encoder_hidden_layer1_ = MatrixXd::Zero(encoder_conv1_output_channel_, encoder_conv1_output_size_);
    encoder_hidden_layer2_ = MatrixXd::Zero(encoder_conv2_output_channel_, encoder_conv2_output_size_);
    encoder_output_ = MatrixXd::Zero(encoder_dim_, 1);

    if (use_encoder_) policy_input_.resize(num_state + encoder_dim_, 1);
    else policy_input_.resize(num_state, 1);

    q_dot_lpf_.setZero();

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  
                    
    if (com_height_ == 0.68){
        q_init_ << 0.0, 0.0, -0.46, 1.04, -0.58, 0.0,
                    0.0, 0.0, -0.46, 1.04, -0.58, 0.0,
                    0.0, 0.0, 0.0,
                    0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                    0.0, 0.0,
                    -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;
    }
    else if (com_height_ == 0.728){
        q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                    0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                    0.0, 0.0, 0.0,
                    0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                    0.0, 0.0,
                    -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;

    }
    else {
        std::cout << "WRONG COM HEIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
        exit(0);
    }

    kp_.setZero();
    kv_.setZero();
    kp_.diagonal() <<   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        6000.0, 10000.0, 10000.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
                        100.0, 100.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0;
    kp_.diagonal() /= 9.0;  
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;
    kv_.diagonal() /= 3.0;

    // Woohyun
    initBias();
    base_lin_vel.setZero();
    base_ang_vel.setZero();
    swing_state_stance_frame_.setZero(13);
    com_state_stance_frame_.setZero(13);
    q_leg_desired_ = q_init_.segment(0, num_actuator_action);
    com_support_current_dot_prev_.setZero(3);

    string cur_path = "/home/cha/catkin_ws/src/tocabi_cc/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
    }

}

Eigen::Vector3d AvatarController::mat2euler(Eigen::Matrix3d mat)
{
    Eigen::Vector3d euler;

    double cy = std::sqrt(mat(2, 2) * mat(2, 2) + mat(1, 2) * mat(1, 2));
    if (cy > std::numeric_limits<double>::epsilon())
    {
        euler(2) = -atan2(mat(0, 1), mat(0, 0));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = -atan2(mat(1, 2), mat(2, 2));
    }
    else
    {
        euler(2) = -atan2(-mat(1, 0), mat(1, 1));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = 0.0;
    }
    return euler;
}

void AvatarController::initBias()
{
    q_bias_.setZero();
    if (~is_on_robot_){
        std::random_device rd;  
        std::mt19937 gen(rd());
        float bias_std = 0.;
        std::uniform_real_distribution<> dis(-bias_std, bias_std);
        q_bias_(2) = dis(gen);
        q_bias_(3) = dis(gen);
        q_bias_(4) = dis(gen);
        q_bias_(8) = dis(gen);
        q_bias_(9) = dis(gen);
        q_bias_(10) = dis(gen);
        // for (int i = 0; i < num_actuator_action; i++){
        //     q_bias_(i) = dis(gen);

        // }
    }
}

void AvatarController::processBias()
{
    for (int i = 0; i < MODEL_DOF; i++){
        q_noise_(i) += q_bias_(i);
    }
}

void AvatarController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    if (is_on_robot_)
    {
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
        q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_dot_lpf_ = q_dot_lpf_;
        }
    }
    else
    {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.00001, 0.00001);
        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
        }
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_vel_noise_ = q_vel_noise_;
            q_dot_lpf_ = q_dot_lpf_;
        }
        q_noise_pre_ = q_noise_;
    }
    time_pre_ = time_cur_;
}


void AvatarController::processObservation() // [linvel, angvel, proj_grav, commands, dof_pos, dof_vel, actions]
{

    int data_idx = 0;

    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);   

    base_lin_vel = q.conjugate()*(rd_cc_.q_dot_virtual_.segment(0,3));
    base_ang_vel = (rd_cc_.q_dot_virtual_.segment(3,3));
    // std::cout <<"global : " << base_ang_vel(0) << ", " << base_ang_vel(1) << ", " << base_ang_vel(2) << std::endl;
    // base_ang_vel = q.conjugate()*base_ang_vel;
    // std::cout << "local : " << base_ang_vel(0) << ", " << base_ang_vel(1) << ", " << base_ang_vel(2) << std::endl;

 
    // for (int i=0; i<6; i++)
    // {
    //     state_cur_(data_idx) = rd_cc_.q_dot_virtual_(i);
    //     data_idx++;
    // }

    for (int i = 0; i < 3; i++){
        state_cur_(data_idx) = base_lin_vel(i);
        data_idx++;
    }

    for (int i = 0; i < 3; i++){
        state_cur_(data_idx) = base_ang_vel(i);
        data_idx++;
    }


    Vector3_t grav, projected_grav, forward_vec;
    grav << 0, 0, -1.;
    forward_vec << 1., 0, 0;
    projected_grav = q.conjugate()*grav;

    // euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());
    // state_cur_(data_idx) = AvatarController::wrap_to_pi(euler_angle_(0));
    // data_idx++;
    // state_cur_(data_idx) = AvatarController::wrap_to_pi(euler_angle_(1));
    // data_idx++;
    // state_cur_(data_idx) = AvatarController::wrap_to_pi(euler_angle_(2));
    // data_idx++;
    state_cur_(data_idx) = q.x();
    data_idx++;
    state_cur_(data_idx) = q.y();
    data_idx++;
    state_cur_(data_idx) = q.z();
    data_idx++;
    state_cur_(data_idx) = q.w();
    data_idx++;

    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_(data_idx) = q_noise_(i) - q_init_(i);
        data_idx++;
    }

    for (int i = 0; i < num_actuator_action; i++)
    {
        if (is_on_robot_)
        {
            state_cur_(data_idx) = q_vel_noise_(i);
        }
        else
        {
            state_cur_(data_idx) = q_vel_noise_(i); //rd_cc_.q_dot_virtual_(i+6);
        }
        data_idx++;
    }

    for (int i = 0; i < num_actuator_action; i++)
    {
        // state_cur_(data_idx) = q_leg_desired_(i) - q_noise_(i);
        state_cur_(data_idx) = q_leg_desired_(i);
        data_idx++;
    }

    // state_cur_(data_idx) = com_state_stance_frame_(0);
    // data_idx++;
    // state_cur_(data_idx) = com_state_stance_frame_(1);
    // data_idx++;
    // state_cur_(data_idx) = com_state_stance_frame_(2);
    // data_idx++;
    // state_cur_(data_idx) = com_state_stance_frame_(3);
    // data_idx++;
    // state_cur_(data_idx) = com_state_stance_frame_(4);
    // data_idx++;
    // state_cur_(data_idx) = com_state_stance_frame_(5);
    // data_idx++;
    // state_cur_(data_idx) = com_state_stance_frame_(6);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(0);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(1);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(2);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(3);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(4);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(5);
    // data_idx++;
    // state_cur_(data_idx) = swing_state_stance_frame_(6);
    // data_idx++;



    state_cur_(data_idx) = cos(float(walking_tick) / float(t_total_(0)) * 2 * M_PI);
    data_idx++;
    state_cur_(data_idx) = sin(float(walking_tick) / float(t_total_(0)) * 2 * M_PI);
    data_idx++;

    state_cur_(data_idx) = step_length_x_(0);
    data_idx++;
    state_cur_(data_idx) = step_length_y_(0);
    data_idx++;
    state_cur_(data_idx) = step_yaw_(0);
    data_idx++;
    state_cur_(data_idx) = t_dsp_seconds(0);
    data_idx++;
    state_cur_(data_idx) = t_ssp_seconds(0);
    data_idx++;
    state_cur_(data_idx) = foot_height_(0);
    data_idx++;

    for (int i = 0; i <num_actuator_action; i++) 
    {
        state_cur_(data_idx) = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
        data_idx++;
    }

    state_buffer_.block(0, 0, num_cur_state*(num_state_skip*num_state_hist-1),1) = state_buffer_.block(num_cur_state, 0, num_cur_state*(num_state_skip*num_state_hist-1),1);
    state_history_.block(0, 0, num_cur_state, history_skip_*history_len_-1) = state_history_.block(0, 1, num_cur_state,history_skip_*history_len_-1);

    MatrixXd tmp = (state_cur_ - state_mean_).array() / state_var_.cwiseSqrt().array();
    for (int i = 0; i < num_cur_state; i++){
        tmp(i) = DyrosMath::minmax_cut(tmp(i), -10., 10.);
    }
    state_buffer_.block(num_cur_state*(num_state_skip*num_state_hist-1), 0, num_cur_state,1) = tmp;
    state_history_.block(0,history_skip_*history_len_-1, num_cur_state, 1) = tmp;

    if (!encoder_initialized){
        for (int i = 0; i < history_skip_*history_len_; i++) 
            state_history_.block(0,i, num_cur_state, 1) = tmp;
        
        encoder_initialized = true;
    }

    // // Internal State First
    // for (int i = 0; i < num_state_hist; i++)
    // {
    //     state_.block(num_cur_internal_state*i, 0, num_cur_internal_state, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_internal_state, 1);
    // }
    // // Action History Second
    // for (int i = 0; i < num_state_hist-1; i++)
    // {
    //     state_.block(num_state_hist*num_cur_internal_state + num_action*i, 0, num_action, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)) + num_cur_internal_state, 0, num_action, 1);
    // }
    for (int i = 0; i < num_state_hist; i++){
        state_.block(num_cur_state*i, 0, num_cur_state, 1) = state_buffer_.block(num_cur_state*(num_state_skip*(i+1)-1), 0, num_cur_state, 1);
    }
    if (use_encoder_){
        for (int i = 0; i < history_len_; i++){
            encoder_input_.block(0, i, num_cur_state, 1) = state_history_.block(0, history_skip_*(i+1)-1, num_cur_state, 1);
       }
    }



}


// ELU VERSION
void AvatarController::feedforwardPolicy()
{
    if (use_encoder_){
        feedforwardEncoder();
        policy_input_.block(0, 0, num_state, 1) = state_;
        policy_input_.block(num_state, 0, encoder_dim_, 1) = encoder_output_;
        
    }
    else policy_input_ = state_;    
    // First hidden layer for policy network
    hidden_layer1_ = policy_net_w0_ * policy_input_ + policy_net_b0_;
    for (int i = 0; i < num_hidden1; i++) 
    {
        if (hidden_layer1_(i) < 0)
            hidden_layer1_(i) = std::exp(hidden_layer1_(i)) - 1.0;
    }

    // Second hidden layer for policy network
    hidden_layer2_ = policy_net_w2_ * hidden_layer1_ + policy_net_b2_;
    for (int i = 0; i < num_hidden2; i++) 
    {
        if (hidden_layer2_(i) < 0)
            hidden_layer2_(i) = std::exp(hidden_layer2_(i)) - 1.0;
    }

    // Output layer for policy network
    rl_action_ = action_net_w_ * hidden_layer2_ + action_net_b_;

    // First hidden layer for value network
    value_hidden_layer1_ = value_net_w0_ * policy_input_ + value_net_b0_;
    for (int i = 0; i < num_hidden1; i++) 
    {
        if (value_hidden_layer1_(i) < 0)
            value_hidden_layer1_(i) = std::exp(value_hidden_layer1_(i)) - 1.0;
    }

    // Second hidden layer for value network
    value_hidden_layer2_ = value_net_w2_ * value_hidden_layer1_ + value_net_b2_;

    for (int i = 0; i < num_hidden2; i++) 
    {
        if (value_hidden_layer2_(i) < 0)
            value_hidden_layer2_(i) = std::exp(value_hidden_layer2_(i)) - 1.0;
    }

    // Output layer for value network
    value_ = (value_net_w_ * value_hidden_layer2_ + value_net_b_)(0);


}

void AvatarController::feedforwardEncoder()
{
    encoder_hidden_layer1_ = conv_layer(encoder_input_, encoder_conv1_w_, encoder_conv1_b_, encoder_conv1_kernel_, encoder_conv1_stride_);
    // encoder_hidden_layer1_ = encoder_hidden_layer1_.cwiseMax(0.);  // ReLU
    encoder_hidden_layer1_ = encoder_hidden_layer1_.array().tanh();  // Tanh

    encoder_hidden_layer2_ = conv_layer(encoder_hidden_layer1_, encoder_conv2_w_, encoder_conv2_b_, encoder_conv2_kernel_, encoder_conv2_stride_);
    // encoder_hidden_layer2_ = encoder_hidden_layer2_.cwiseMax(0.);  // ReLU
    encoder_hidden_layer2_ = encoder_hidden_layer2_.array().tanh();  // Tanh

    Eigen::MatrixXd flattened = Eigen::Map<const Eigen::MatrixXd>(encoder_hidden_layer2_.data(), encoder_hidden_layer2_.size(), 1);
    encoder_output_ = (encoder_fc_w_ * flattened) + encoder_fc_b_;
    encoder_output_ = encoder_output_.array().tanh();
}

Eigen::MatrixXd AvatarController::conv_layer(const MatrixXd &input, const MatrixXd &weights, const MatrixXd &biases, int kernel_size, int stride)
{
    // Implement 1D convolution manually
    // Here, input is of size (d, T), weights of size (out_channels, in_channels * kernel_size)
    int out_channels = biases.rows();
    int T_out = (input.cols() - kernel_size) / stride + 1;
    Eigen::MatrixXd output(out_channels, T_out);

    for (int i = 0; i < out_channels; ++i) {
        for (int t = 0; t < T_out; ++t) {
            int start_idx = t * stride;
            Eigen::VectorXd segment = Eigen::Map<const Eigen::VectorXd>(
                input.data() + start_idx * input.rows(), input.rows() * kernel_size);
            output(i, t) = segment.dot(weights.row(i)) + biases(i, 0);
        }
    }

    return output;
}

void AvatarController::loadEncoderNetwork()
{
    string cur_path = "/home/cha/catkin_ws/src/tocabi_cc/";

    if (is_on_robot_)
    {
        cur_path = "/home/dyros/catkin_ws/src/tocabi_cc/";
    }
    base_path = loadPathFromConfig(cur_path + "weight_directory.txt");

    std::ifstream file[6];
    file[0].open(base_path + "encoder/conv1_weight.txt", std::ios::in);
    file[1].open(base_path + "encoder/conv1_bias.txt", std::ios::in);
    file[2].open(base_path + "encoder/conv2_weight.txt", std::ios::in);
    file[3].open(base_path + "encoder/conv2_bias.txt", std::ios::in);
    file[4].open(base_path + "encoder/fc2_weight.txt", std::ios::in);
    file[5].open(base_path + "encoder/fc2_bias.txt", std::ios::in);

    if (!file[0].is_open()) {
        std::cout << "Cannot find the weight file" << std::endl;
    }

    float temp;
    int row = 0;
    int col = 0;

    // Load conv1 weights
    while (!file[0].eof() && row != encoder_conv1_w_.rows())
    {
        file[0] >> temp;
        encoder_conv1_w_(row, col) = temp;
        col++;
        if (col == encoder_conv1_w_.cols())
        {
            col = 0;
            row++;
        }
    }

    // Load conv1 biases
    row = 0;
    col = 0;
    while (!file[1].eof() && row != encoder_conv1_b_.rows())
    {
        file[1] >> temp;
        encoder_conv1_b_(row, col) = temp;
        col++;
        if (col == encoder_conv1_b_.cols())
        {
            col = 0;
            row++;
        }
    }

    // Load conv2 weights
    row = 0;
    col = 0;
    while (!file[2].eof() && row != encoder_conv2_w_.rows())
    {
        file[2] >> temp;
        encoder_conv2_w_(row, col) = temp;
        col++;
        if (col == encoder_conv2_w_.cols())
        {
            col = 0;
            row++;
        }
    }

    // Load conv2 biases
    row = 0;
    col = 0;
    while (!file[3].eof() && row != encoder_conv2_b_.rows())
    {
        file[3] >> temp;
        encoder_conv2_b_(row, col) = temp;
        col++;
        if (col == encoder_conv2_b_.cols())
        {
            col = 0;
            row++;
        }
    }

    // Load fully connected layer weights
    row = 0;
    col = 0;
    while (!file[4].eof() && row != encoder_fc_w_.rows())
    {
        file[4] >> temp;
        encoder_fc_w_(row, col) = temp;
        col++;
        if (col == encoder_fc_w_.cols())
        {
            col = 0;
            row++;
        }
    }

    // Load fully connected layer biases
    row = 0;
    col = 0;
    while (!file[5].eof() && row != encoder_fc_b_.rows())
    {
        file[5] >> temp;
        encoder_fc_b_(row, col) = temp;
        col++;
        if (col == encoder_fc_b_.cols())
        {
            col = 0;
            row++;
        }
    }
}



void AvatarController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{   

    last_buttons_0.push_back(joy->buttons[0]);
    last_buttons_1.push_back(joy->buttons[1]);
    last_buttons_2.push_back(joy->buttons[2]);
    last_buttons_3.push_back(joy->buttons[3]);
    last_buttons_7.push_back(joy->axes[7]);
    // joy_length_previous_vec.push_back(joy_length);

    if (last_buttons_0.size() > 10) {
        last_buttons_0.erase(last_buttons_0.begin());
    }
    if (last_buttons_1.size() > 10) {
        last_buttons_1.erase(last_buttons_1.begin()); 
    }
    if (last_buttons_2.size() > 10) {
        last_buttons_2.erase(last_buttons_2.begin());
    }
    if (last_buttons_3.size() > 10) {
        last_buttons_3.erase(last_buttons_3.begin()); 
    }
    if (last_buttons_7.size() > 10) {
        last_buttons_7.erase(last_buttons_7.begin());
    }
    if(iter_x_l>1){
        joy_length_previous = joy_length_command;
        iter_x_l =0;
    }
    if(iter_x_r>1){
        joy_length_previous = joy_length_command;
        iter_x_r =0;
    }
    if(iter_end_x>3){
        joy_length_previous = joy_length_command;
        iter_end_x =0;
    }



    if(iter_y_l_ls>1){
        joy_length_y_l_previous = joy_length_y_l_command;
        iter_y_l_ls =0;
    }
    if(iter_y_l_rs>2){
        joy_length_y_l_previous = joy_length_y_l_command;
        iter_y_l_rs =0;
    }
    if(iter_end_y_l_ls>1){
        joy_length_y_l_previous = joy_length_y_l_command;
        iter_end_y_l_ls =0;
    }
    if(iter_end_y_l_rs>2){
        joy_length_y_l_previous = joy_length_y_l_command;
        iter_end_y_l_rs =0;
    }



    if(iter_y_r_ls>2){
        joy_length_y_r_previous = joy_length_y_r_command;
        iter_y_r_ls =0;
    }
    if(iter_y_r_rs>1){
        joy_length_y_r_previous = joy_length_y_r_command;
        iter_y_r_rs =0;
    }
    if(iter_end_y_r_ls>2){
        joy_length_y_r_previous = joy_length_y_r_command;
        iter_end_y_r_ls =0;
    }
    if(iter_end_y_r_rs>1){
        joy_length_y_r_previous = joy_length_y_r_command;
        iter_end_y_r_rs =0;
    }


    
    if(fabs(joy->axes[0]) < 0.1 && fabs(joy->axes[1]) < 0.1){ 
        
        joy_length =0.0;
        joy_length_y_r_temp =0.21;
        joy_length_y_l_temp =0.21;
        // if(joy_length = 0.0){
        //     joy_length_previous =0.0;
        // }
        // Lcommand_step_length_x_ = joy_length_l;
        Lcommand_step_length_x_ = 0.;
        // std::cout<<joy_length_r<<std::endl;
        // Lcommand_step_length_y_ = joy_length_y_l;
        Lcommand_step_length_y_ = 0.21;
        // Lcommand_step_yaw_ = joy_yaw_l;
        // std::cout << "joy yaw l :" << joy_yaw_l << std::endl;
        // Lcommand_step_yaw_ = 0.;
       // Lcommand_t_dsp_ = 0.2;
        // Lcommand_t_ssp_ = 1.0;
        // Lcommand_foot_height_ = 0.1; 
        // Rcommand_step_length_x_ = joy_length_r;
        Rcommand_step_length_x_ = 0.;
        // Rcommand_step_length_y_ = joy_length_y_r;
        Rcommand_step_length_y_ = 0.21;
        // Rcommand_step_yaw_ = joy_yaw_r;
        // std::cout << "joy yaw r :" << joy_yaw_r << std::endl;
        // Rcommand_step_yaw_ = 0.;
        // Rcommand_t_dsp_ = 0.2;
        // Rcommand_t_ssp_ = 1.0;
        // Rcommand_foot_height_ = 0.1;

    }else{

        joy_x = joy->axes[1];
        joy_y = joy->axes[0];
        norm = sqrt(pow(joy_x,2) + pow(joy_y,2));

        joy_x /=norm;
        joy_y /=norm;

        
        if(joy_x > 0.0){
            joy_length = joy_length_temp*joy_x;
            // joy_length_l =joy_length_temp*joy_x;
            // joy_length_r =joy_length_temp*joy_x;
            // std::cout<<joy_length<<std::endl;
            Lcommand_step_length_x_ = joy_length;
            Rcommand_step_length_x_ = joy_length;
        }else if(joy_x < 0.0){
            joy_length = DyrosMath::minmax_cut(joy_length_temp*joy_x, -0.3, 0.0);
            // joy_length_l =joy_length_temp*-joy_x;
            // joy_length_r =joy_length_temp*-joy_x;
                // std::cout<<joy_length<<std::endl;
            Lcommand_step_length_x_ = joy_length;
            Rcommand_step_length_x_ = joy_length;
        }

        if(joy_y > 0.0){

            joy_length_y_l_temp = 0.21 + (joy_length_y_temp - 0.21)*joy_y; 
            joy_length_y_r_temp =0.21;
            Lcommand_step_length_y_= joy_length_y_l_temp;
            Rcommand_step_length_y_= joy_length_y_r_temp;
            // std::cout<<joy_length_y_r<<std::endl;
            // std::cout<<joy_y<<std::endl;
            // std::cout<<Lcommand_step_length_y_<<std::endl;
        }else if(joy_y < 0.0){

            joy_length_y_r_temp = 0.21 +  (joy_length_y_temp - 0.21)*-joy_y;
            joy_length_y_l_temp = 0.21;
            Lcommand_step_length_y_= joy_length_y_l_temp;             
            Rcommand_step_length_y_= joy_length_y_r_temp;
            // std::cout<<Rcommand_step_length_y_<<std::endl;
        }
    }
    //  std::cout<<joy_length_y_r<<std::endl;
    

    if(joy->axes[7] == -1.0 && Lcommand_t_dsp_ < 0.2 && Lcommand_t_ssp_<1.0){ //long step
        if(last_buttons_7[last_buttons_7.size()-2]==0.0){
            Lcommand_t_dsp_ = DyrosMath::minmax_cut(Lcommand_t_dsp_+0.01, 0.02, 0.2);
            Lcommand_t_ssp_ = DyrosMath::minmax_cut(Lcommand_t_ssp_+0.05, 0.5, 1.0);
            Rcommand_t_dsp_ = Lcommand_t_dsp_;
            Rcommand_t_ssp_ = Lcommand_t_ssp_;
            ROS_INFO("%f",Lcommand_t_dsp_);
            ROS_INFO("%f",Lcommand_t_ssp_);
        }

    }else if(joy->axes[7] == 1.0 && Lcommand_t_dsp_> 0.02 && Lcommand_t_ssp_> 0.5){//short step
        if(last_buttons_7[last_buttons_7.size()-2]==0.0){
            Lcommand_t_dsp_ = DyrosMath::minmax_cut(Lcommand_t_dsp_-0.01, 0.02, 0.2);
            Lcommand_t_ssp_ = DyrosMath::minmax_cut(Lcommand_t_ssp_-0.05, 0.5, 1.0);
            Rcommand_t_dsp_ = Lcommand_t_dsp_;
            Rcommand_t_ssp_ = Lcommand_t_ssp_;
            ROS_INFO("%f",Lcommand_t_dsp_);
            ROS_INFO("%f",Lcommand_t_ssp_);
            }
    }else{
        Lcommand_t_dsp_ = Lcommand_t_dsp_;
        Lcommand_t_ssp_ = Lcommand_t_ssp_;

        Rcommand_t_dsp_ = Lcommand_t_dsp_;
        Rcommand_t_ssp_ = Lcommand_t_ssp_;
    }
    
    if(joy->buttons[1] == 1 && joy_length_temp < 0.5 && joy_length_y_temp < 0.5){ 
        if(last_buttons_1[last_buttons_1.size()-2]==0){

            joy_length_temp+=0.06;//보폭
            joy_length_y_temp +=0.04;
            joy_length_temp = DyrosMath::minmax_cut(joy_length_temp, 0.1, 0.5);
            joy_length_y_temp = DyrosMath::minmax_cut(joy_length_y_temp, 0.25, 0.5);

            ROS_INFO("step_length_x_ : %f",joy_length_temp);
            ROS_INFO("step_length_y_ : %f",joy_length_y_temp);
        }            
    }
    if(joy->buttons[0] == 1 &&  joy_length_temp > 0.1 && joy_length_y_temp > 0.25){ 
        if(last_buttons_0[last_buttons_0.size()-2]==0){
            // 0.06 0.12 0.18 0.24 0.30 0.36
            // 0.25 0.29 0.33 0.37 0.41 0.45         

            joy_length_temp-=0.06;//보폭
            joy_length_y_temp-=0.04;//보폭
            joy_length_temp = DyrosMath::minmax_cut(joy_length_temp, 0.1, 0.5);
            joy_length_y_temp = DyrosMath::minmax_cut(joy_length_y_temp, 0.25, 0.5);
            ROS_INFO("step_length_x_ : %f",joy_length_temp);
            ROS_INFO("step_length_y_ : %f",joy_length_y_temp);
            
        }            
    }

    if(joy->buttons[3]==1 && Lcommand_foot_height_<0.2){
        if(last_buttons_3[last_buttons_3.size()-2]==0){
            joy_height= DyrosMath::minmax_cut(Lcommand_foot_height_+0.02, 0.05, 0.2);
            Lcommand_foot_height_= Rcommand_foot_height_ = joy_height;
            ROS_INFO("%f",joy_height);
        }

    }else if(joy->buttons[2]==1 && Lcommand_foot_height_>0.05){
        if(last_buttons_2[last_buttons_2.size()-2]==0){
            joy_height= DyrosMath::minmax_cut(Lcommand_foot_height_-0.02, 0.05, 0.2);
        Lcommand_foot_height_= Rcommand_foot_height_ = joy_height;
        ROS_INFO("%f",joy_height);
        }
    }else{
        Lcommand_foot_height_= joy_height;
        Rcommand_foot_height_= joy_height;
    }

    if(joy->buttons[4] == 1){
        joy_yaw_l_command =0.3;
        joy_yaw_r_command =0.0;

        Lcommand_step_yaw_ = joy_yaw_l_command;           
        Rcommand_step_yaw_ = joy_yaw_r_command;
    }
    if(joy->buttons[5] == 1){
        joy_yaw_l_command =0.0;
        joy_yaw_r_command =0.3;

        Lcommand_step_yaw_ = joy_yaw_l_command;           
        Rcommand_step_yaw_ = joy_yaw_r_command;

    }
    if(joy->buttons[5] != 1 && joy->buttons[4] != 1){
        joy_yaw_l_command =0.0;
        joy_yaw_r_command =0.0;
        
        Lcommand_step_yaw_ = joy_yaw_l_command;           
        Rcommand_step_yaw_ = joy_yaw_r_command;
    }

    
    // target_vel_x_ = DyrosMath::minmax_cut(0.5*joy->axes[1], -0.2, 0.5);
    // target_vel_y_ = DyrosMath::minmax_cut(0.5*joy->axes[0], -0.2, 0.2);
    // std::cout << "Rcommand_step_yaw_ :" << Rcommand_step_yaw_ << std::endl;
}


std::string AvatarController::loadPathFromConfig(const std::string &config_file)
{
    std::cout << "LOAD WEIGHT!!" << std::endl;
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open configuration file: " + config_file);
    }

    std::string line, key, value;
    while (std::getline(file, line))
    {
        std::istringstream line_stream(line);
        if (std::getline(line_stream, key, '=') && std::getline(line_stream, value))
        {
            if (key == "weights_path")
            {
                file.close();
                return value; // Return the weights path
            }
        }
    }

    file.close();
    throw std::runtime_error("weights_path not found in configuration file.");
}

void AvatarController::updateInitialState()
{

    pelv_rpy_current_WH.setZero();
    pelv_rpy_current_WH = DyrosMath::rot2Euler(rd_cc_.link_[Pelvis].rotm); //ZYX multiply

    pelv_yaw_rot_current_from_global_WH = DyrosMath::rotateWithZ(pelv_rpy_current_WH(2));
    pelv_yaw_rot_current_from_global_WH.linear() = rd_cc_.link_[Pelvis].rotm;

    rfoot_rpy_current_.setZero();
    lfoot_rpy_current_.setZero();
    rfoot_rpy_current_ = DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm);
    lfoot_rpy_current_ = DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm);

    rfoot_roll_rot_ = DyrosMath::rotateWithX(rfoot_rpy_current_(0));
    lfoot_roll_rot_ = DyrosMath::rotateWithX(lfoot_rpy_current_(0));
    rfoot_pitch_rot_ = DyrosMath::rotateWithY(rfoot_rpy_current_(1));
    lfoot_pitch_rot_ = DyrosMath::rotateWithY(lfoot_rpy_current_(1));
    rfoot_yaw_rot_ = DyrosMath::rotateWithZ(rfoot_rpy_current_(2));
    lfoot_yaw_rot_ = DyrosMath::rotateWithZ(lfoot_rpy_current_(2));


    if (foot_step_(0, 6) == 0) //right foot support
    {
        supportfoot_global_init_.translation() = rd_cc_.link_[Right_Foot].xpos;
        supportfoot_global_init_.linear() = rd_cc_.link_[Right_Foot].rotm;
    }
    else if (foot_step_(0, 6) == 1)
    {
        supportfoot_global_init_.translation() = rd_cc_.link_[Left_Foot].xpos;
        supportfoot_global_init_.linear() = rd_cc_.link_[Left_Foot].rotm;
    }

    // yaw only
    supportfoot_global_init_yaw_ = supportfoot_global_init_;
    supportfoot_global_init_yaw_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(supportfoot_global_init_.linear())(2));

    Eigen::Isometry3d ref_frame;
    ref_frame = supportfoot_global_init_yaw_;
    pelv_support_init_yaw_.translation() =DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(ref_frame) , rd_cc_.link_[Pelvis].xpos);
    pelv_support_init_yaw_.linear() = ref_frame.linear().transpose() *  rd_cc_.link_[Pelvis].rotm;
    com_support_init_yaw_ = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(ref_frame), rd_cc_.link_[COM_id].xpos);
    com_support_init_dot_yaw_ = ref_frame.linear().transpose() * rd_cc_.link_[COM_id].v;
    pelv_support_euler_init_yaw_ = DyrosMath::rot2Euler(pelv_support_init_yaw_.linear());
    lfoot_global_init_.translation() = rd_cc_.link_[Left_Foot].xpos;
    lfoot_global_init_.linear() = rd_cc_.link_[Left_Foot].rotm;
    rfoot_global_init_.translation() = rd_cc_.link_[Right_Foot].xpos;
    rfoot_global_init_.linear() = rd_cc_.link_[Right_Foot].rotm;

    lfoot_support_init_yaw_ = DyrosMath::multiplyIsometry3d(DyrosMath::inverseIsometry3d(ref_frame), lfoot_global_init_);
    rfoot_support_init_yaw_ = DyrosMath::multiplyIsometry3d(DyrosMath::inverseIsometry3d(ref_frame), rfoot_global_init_);
    rfoot_support_euler_init_yaw_ = DyrosMath::rot2Euler(rfoot_support_init_yaw_.linear());
    lfoot_support_euler_init_yaw_ = DyrosMath::rot2Euler(lfoot_support_init_yaw_.linear());

}


// NO DELAYED COMMANDS
// /*
void AvatarController::updateFootstepCommand(){



    if (walking_tick == 0){


        for (int step = 0; step < number_of_foot_step; step++){
            if (step == 0) phase_indicator_(step) = first_stance_foot_;
            else phase_indicator_(step) = 1-phase_indicator_(step-1);

            step_length_x_(step) = phase_indicator_(step)*Lcommand_step_length_x_ + (1-phase_indicator_(step))*Rcommand_step_length_x_;
            step_length_y_(step) = (2*phase_indicator_(step) - 1) *(phase_indicator_(step)*Lcommand_step_length_y_ + (1-phase_indicator_(step))*Rcommand_step_length_y_);
            step_yaw_(step) = (2*phase_indicator_(step) - 1) * (phase_indicator_(step)*Lcommand_step_yaw_ + (1-phase_indicator_(step))*Rcommand_step_yaw_);
            foot_height_(step) = phase_indicator_(step)*Lcommand_foot_height_ + (1-phase_indicator_(step))*Rcommand_foot_height_;
            t_dsp_(step) = std::floor((phase_indicator_(step)*Lcommand_t_dsp_ + (1-phase_indicator_(step))*Rcommand_t_dsp_) * hz_);
            t_dsp_seconds(step) = phase_indicator_(step)*Lcommand_t_dsp_ + (1-phase_indicator_(step))*Rcommand_t_dsp_;
            t_ssp_(step) = std::floor((phase_indicator_(step)*Lcommand_t_ssp_ + (1-phase_indicator_(step))*Rcommand_t_ssp_ )* hz_);
            t_ssp_seconds(step) = phase_indicator_(step)*Lcommand_t_ssp_ + (1-phase_indicator_(step))*Rcommand_t_ssp_;
            t_total_(step) = 2*t_dsp_(step) + t_ssp_(step);

            
        }

        calculateFootStepTotal();



        getRobotState();



        updateInitialState();



        getZmpTrajectory();



        resetPreviewState();



    }

    else if (walking_tick > t_total_(0)){

        std::cout << "Foot Position error : " << sqrt(pow(swing_state_stance_frame_(0) - step_length_x_(0), 2) + pow(swing_state_stance_frame_(1) - step_length_y_(0), 2)) << " [m]" << std::endl;
        std::cout << ">> X error : " << sqrt(pow(swing_state_stance_frame_(0) - step_length_x_(0), 2)) << " [m]" << std::endl;
        std::cout << ">> Y error : " << sqrt(pow(swing_state_stance_frame_(1) - step_length_y_(0), 2)) << " [m]" << std::endl;
        Eigen::Quaterniond q1(swing_state_stance_frame_(6), swing_state_stance_frame_(3), swing_state_stance_frame_(4), swing_state_stance_frame_(5));
        double swing_yaw = std::atan2(2.0 * (q1.w() * q1.z() + q1.x() * q1.y()),
                             1.0 - 2.0 * (q1.y() * q1.y() + q1.z() * q1.z()));
        std::cout << "Foot Yaw error : " << sqrt(pow(AvatarController::wrap_to_pi(swing_yaw - step_yaw_(0)), 2)) << " [rad]" << std::endl;
        

        x_error = (step_length_x_(0) - swing_state_stance_frame_(0)) ;
        y_error = (step_length_y_(0) - swing_state_stance_frame_(1)) ;
        yaw_error = (AvatarController::wrap_to_pi(swing_yaw - step_yaw_(0))) ;

        step_length_x_.segment(0,number_of_foot_step-1) = step_length_x_.segment(1,number_of_foot_step-1);

        step_length_y_.segment(0,number_of_foot_step-1) = step_length_y_.segment(1,number_of_foot_step-1);

        step_yaw_.segment(0,number_of_foot_step-1) = step_yaw_.segment(1,number_of_foot_step-1);

        t_dsp_.segment(0,number_of_foot_step-1) = t_dsp_.segment(1,number_of_foot_step-1);
        t_dsp_seconds.segment(0,number_of_foot_step-1) = t_dsp_seconds.segment(1,number_of_foot_step-1);

        t_ssp_.segment(0,number_of_foot_step-1) = t_ssp_.segment(1,number_of_foot_step-1);
        t_ssp_seconds.segment(0,number_of_foot_step-1) = t_ssp_seconds.segment(1,number_of_foot_step-1);

        t_total_.segment(0, number_of_foot_step-1) = t_total_.segment(1, number_of_foot_step-1);

        foot_height_.segment(0,number_of_foot_step-1) = foot_height_.segment(1,number_of_foot_step-1);

        phase_indicator_.segment(0,number_of_foot_step-1) = phase_indicator_.segment(1,number_of_foot_step-1);

        phase_indicator_(number_of_foot_step-1) = 1-phase_indicator_(number_of_foot_step-2);

        int step = number_of_foot_step-1;
        step_length_x_(step) = phase_indicator_(step)*(Lcommand_step_length_x_) + (1-phase_indicator_(step))*(Rcommand_step_length_x_);
        step_length_y_(step) = (2*phase_indicator_(step) - 1) *(phase_indicator_(step)*(Lcommand_step_length_y_) + (1-phase_indicator_(step))*(Rcommand_step_length_y_));
        step_yaw_(step) = (2*phase_indicator_(step) - 1) * (phase_indicator_(step)*(Lcommand_step_yaw_) + (1-phase_indicator_(step))*(Rcommand_step_yaw_));
        foot_height_(step) = phase_indicator_(step)*Lcommand_foot_height_ + (1-phase_indicator_(step))*Rcommand_foot_height_;
        t_dsp_(step) = std::floor((phase_indicator_(step)*Lcommand_t_dsp_ + (1-phase_indicator_(step))*Rcommand_t_dsp_) * hz_);
        t_dsp_seconds(step) = phase_indicator_(step)*Lcommand_t_dsp_ + (1-phase_indicator_(step))*Rcommand_t_dsp_;
        t_ssp_(step) = std::floor((phase_indicator_(step)*Lcommand_t_ssp_ + (1-phase_indicator_(step))*Rcommand_t_ssp_ )* hz_);
        t_ssp_seconds(step) = phase_indicator_(step)*Lcommand_t_ssp_ + (1-phase_indicator_(step))*Rcommand_t_ssp_;
        t_total_(step) = 2*t_dsp_(step) + t_ssp_(step);


        walking_tick = 0;
        current_step_number++;

        calculateFootStepTotal();



        getRobotState();



        updateInitialState();



        getZmpTrajectory();



        resetPreviewState();



    }

}



void AvatarController::getRobotState()
{

    pelv_rpy_current_WH.setZero();
    pelv_rpy_current_WH = DyrosMath::rot2Euler(rd_cc_.link_[Pelvis].rotm); //ZYX multiply

    R_angle = pelv_rpy_current_WH(0);
    P_angle = pelv_rpy_current_WH(1);
    pelv_yaw_rot_current_from_global_WH = DyrosMath::rotateWithZ(pelv_rpy_current_WH(2));
    pelv_yaw_rot_current_from_global_WH.linear() = rd_cc_.link_[Pelvis].rotm;

    rfoot_rpy_current_.setZero();
    lfoot_rpy_current_.setZero();
    rfoot_rpy_current_ = DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm);
    lfoot_rpy_current_ = DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm);

    rfoot_roll_rot_ = DyrosMath::rotateWithX(rfoot_rpy_current_(0));
    lfoot_roll_rot_ = DyrosMath::rotateWithX(lfoot_rpy_current_(0));
    rfoot_pitch_rot_ = DyrosMath::rotateWithY(rfoot_rpy_current_(1));
    lfoot_pitch_rot_ = DyrosMath::rotateWithY(lfoot_rpy_current_(1));

    pelv_global_current_.translation() = rd_cc_.link_[Pelvis].xpos;
    pelv_global_current_.linear() = rd_cc_.link_[Pelvis].rotm;
    lfoot_global_current_.translation() = rd_cc_.link_[Left_Foot].xpos;
    lfoot_global_current_.linear() = rd_cc_.link_[Left_Foot].rotm;
    rfoot_global_current_.translation() = rd_cc_.link_[Right_Foot].xpos;
    rfoot_global_current_.linear() = rd_cc_.link_[Right_Foot].rotm;
    com_global_current_ = rd_cc_.link_[COM_id].xpos;
    com_global_current_dot_prev_ = com_global_current_dot_;
    com_global_current_dot_ = rd_cc_.link_[COM_id].v;

    double support_foot_flag = foot_step_(0, 6);
    if (support_foot_flag == 0)
    {
        supportfoot_global_current_.translation() = rd_cc_.link_[Right_Foot].xpos;
        supportfoot_global_current_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(rd_cc_.link_[Right_Foot].rotm)(2));
    }
    else if (support_foot_flag == 1)
    {
        supportfoot_global_current_.translation() = rd_cc_.link_[Left_Foot].xpos;
        supportfoot_global_current_.linear() = DyrosMath::rotateWithZ(DyrosMath::rot2Euler(rd_cc_.link_[Left_Foot].rotm)(2));
    }

    pelv_support_current_ = DyrosMath::inverseIsometry3d(supportfoot_global_current_) * pelv_global_current_;
    lfoot_support_current_ = DyrosMath::inverseIsometry3d(supportfoot_global_current_) * lfoot_global_current_;
    rfoot_support_current_ = DyrosMath::inverseIsometry3d(supportfoot_global_current_) * rfoot_global_current_;

    com_support_current_ = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(supportfoot_global_current_), com_global_current_);
    com_support_current_dot_ = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(supportfoot_global_current_), com_global_current_dot_);
    // std::cout << "Support foot is : " << ((phase_indicator_(0)) ? "right" : "left") << std::endl;
    // std::cout << "support foot global pos : " << supportfoot_global_init_.translation().transpose() << std::endl;
    // std::cout << "Left foot global pos : " << rd_.link_[Left_Foot].xpos.transpose() << std::endl;
    // std::cout << "Right foot global pos : " << rd_.link_[Right_Foot].xpos.transpose() << std::endl;
    swing_state_stance_frame_.segment(0,3) = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(supportfoot_global_current_), phase_indicator_(0)*rd_cc_.link_[Left_Foot].xpos + (1-phase_indicator_(0))*rd_cc_.link_[Right_Foot].xpos); // Compute swing foot state from init support foot
    Eigen::Quaterniond swing_quat(phase_indicator_(0)*(supportfoot_global_current_.linear().transpose() * rd_cc_.link_[Left_Foot].rotm) + (1-phase_indicator_(0))*(supportfoot_global_current_.linear().transpose() * rd_cc_.link_[Right_Foot].rotm));
    swing_state_stance_frame_(3) = swing_quat.x();
    swing_state_stance_frame_(4) = swing_quat.y();
    swing_state_stance_frame_(5) = swing_quat.z();
    swing_state_stance_frame_(6) = swing_quat.w();

    com_state_stance_frame_.segment(0,3) = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(supportfoot_global_current_), rd_cc_.link_[COM_id].xpos); // Compute swing foot state from init support foot
    Eigen::Quaterniond com_quat(supportfoot_global_current_.linear().transpose() * rd_cc_.link_[COM_id].rotm);
    com_state_stance_frame_(3) = com_quat.x();
    com_state_stance_frame_(4) = com_quat.y();
    com_state_stance_frame_(5) = com_quat.z();
    com_state_stance_frame_(6) = com_quat.w();

    if (!ideal_preview){
        x_preview_(0) = com_support_current_(0);
        y_preview_(0) = com_support_current_(1);
        
        x_preview_(1) = com_support_current_dot_(0);
        y_preview_(1) = com_support_current_dot_(1);
    
        x_preview_(2) = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(supportfoot_global_current_), com_global_current_dot_ - com_global_current_dot_prev_)(0) * hz_;
        y_preview_(2) = DyrosMath::multiplyIsometry3dVector3d(DyrosMath::inverseIsometry3d(supportfoot_global_current_), com_global_current_dot_ - com_global_current_dot_prev_)(1) * hz_;
    }

}

void AvatarController::calculateFootStepTotal()
{

    foot_step_.resize(number_of_foot_step, 7);
    foot_step_.setZero();
    foot_step_support_frame_.resize(number_of_foot_step, 7);
    foot_step_support_frame_.setZero();
    
    // foot_step_ is foothold command in that step's stance foot
    // foot_step_support_frame_ is foothold command in the first stance foot frame
    foot_step_(0,0) = step_length_x_(0);
    foot_step_(0,1) = step_length_y_(0);
    foot_step_(0,5) = step_yaw_(0);
    foot_step_(0,6) = 1-phase_indicator_(0);
    foot_step_support_frame_(0, 0) = step_length_x_(0);
    foot_step_support_frame_(0, 1) = step_length_y_(0);
    foot_step_support_frame_(0, 5) = step_yaw_(0);
    foot_step_support_frame_(0, 6) = 1-phase_indicator_(0);

    for (int i = 1; i < number_of_foot_step; i++){
        foot_step_(i,0) = step_length_x_(i);
        foot_step_(i,1) = step_length_y_(i);
        foot_step_(i,5) = step_yaw_(i);
        foot_step_(i,6) = 1-phase_indicator_(i);
        
        foot_step_support_frame_(i, 0) = foot_step_support_frame_(i-1, 0) + cos(foot_step_support_frame_(i-1, 5)) * step_length_x_(i) - sin(foot_step_support_frame_(i-1, 5)) * step_length_y_(i);
        foot_step_support_frame_(i, 1) = foot_step_support_frame_(i-1, 1) + sin(foot_step_support_frame_(i-1, 5)) * step_length_x_(i) + cos(foot_step_support_frame_(i-1, 5)) * step_length_y_(i);
        foot_step_support_frame_(i, 5) = foot_step_support_frame_(i-1, 5) + step_yaw_(i);
        foot_step_support_frame_(i, 0) += phase_indicator_(i)*zmp_offset*sin(foot_step_support_frame_(i, 5)) - (1 - phase_indicator_(i))*zmp_offset*sin(foot_step_support_frame_(i, 5));
        foot_step_support_frame_(i, 1) += -phase_indicator_(i)*zmp_offset*cos(foot_step_support_frame_(i, 5)) + (1 - phase_indicator_(i))*zmp_offset*cos(foot_step_support_frame_(i, 5));
        foot_step_support_frame_(i, 6) = 1-phase_indicator_(i);
    }





}


void AvatarController::addZmpOffset()
{

    foot_step_support_frame_offset_ = foot_step_support_frame_;
    foot_step_offset_ = foot_step_;
    // ZMP OFFSET
    for (int i = 0; i < number_of_foot_step; i++){
        foot_step_offset_(i, 0) += phase_indicator_(i)*zmp_offset*sin(foot_step_(i, 5)) - (1 - phase_indicator_(i))*zmp_offset*sin(foot_step_(i, 5));
        foot_step_offset_(i, 1) += -phase_indicator_(i)*zmp_offset*cos(foot_step_(i, 5)) + (1 - phase_indicator_(i))*zmp_offset*cos(foot_step_(i, 5));

        foot_step_support_frame_offset_(i, 0) += phase_indicator_(i)*zmp_offset*sin(foot_step_support_frame_(i, 5)) - (1 - phase_indicator_(i))*zmp_offset*sin(foot_step_support_frame_(i, 5));
        foot_step_support_frame_offset_(i, 1) += -phase_indicator_(i)*zmp_offset*cos(foot_step_support_frame_(i, 5)) + (1 - phase_indicator_(i))*zmp_offset*cos(foot_step_support_frame_(i, 5));
    }
}


void AvatarController::getZmpTrajectory()
{
    unsigned int norm_size = 0;

    norm_size = 4.0*hz_ ; // compute zmp over the three planned steps
    addZmpOffset(); 

    zmpGenerator(norm_size);

}


void AvatarController::zmpGenerator(const unsigned int norm_size)
{
    /*
    Goal
    ----------
    -> To position the ZMP at the center of the foot sole according to the footstep planning to prevent the robot from falling during walking.

    Parameters
    ----------
    -> norm_size : The size of the previewed vector for the ZMP reference.

    -> planning_step_num : The number of footsteps to be predicted.

    Returns
    -------
    -> ref_zmp_ : The ZMP reference vector calculated based on the foot sole plan (size: 2 x norm_size).
    */

    ref_zmp_.setZero(norm_size, 2);
    ref_zmp_thread3.setZero(norm_size, 2);
    ref_com_yaw_.setZero(norm_size);
    ref_com_yawvel_.setZero(norm_size);

    Eigen::VectorXd temp_px;
    Eigen::VectorXd temp_py;
    Eigen::VectorXd temp_yaw;
    Eigen::VectorXd temp_yawvel;

    unsigned int index = 0;

  
    for (unsigned int i = 0; i < number_of_foot_step; i++)
    {   
        onestepZmp(i, temp_px, temp_py, temp_yaw, temp_yawvel); // save 1-step zmp into temp px, py
        ref_zmp_.block(index, 0, t_total_(i), 1) = temp_px; 
        ref_zmp_.block(index, 1, t_total_(i), 1) = temp_py;
        ref_com_yaw_.segment(index, t_total_(i)) = temp_yaw;
        ref_com_yawvel_.segment(index, t_total_(i)) = temp_yawvel;

        index = index + t_total_(i);                                                          
    }
    if (t_total_.sum() < norm_size){
        for (int i = t_total_.sum(); i < norm_size; i++){
            ref_zmp_(i, 0) = ref_zmp_(i-1, 0);
            ref_zmp_(i, 1) = ref_zmp_(i-1, 1);
            ref_com_yaw_(i) = ref_com_yaw_(i-1);
            ref_com_yawvel_(i) = ref_com_yawvel_(i-1);
        }
    }

}

void AvatarController::onestepZmp(unsigned int current_step_number, Eigen::VectorXd &temp_px, Eigen::VectorXd &temp_py, Eigen::VectorXd &temp_yaw, Eigen::VectorXd &temp_yawvel) // CoM Yaw as well.
{
    temp_px.setZero(t_total_(current_step_number));  
    temp_py.setZero(t_total_(current_step_number));
    temp_yaw.setZero(t_total_(current_step_number));
    temp_yawvel.setZero(t_total_(current_step_number));

    double v0_x_dsp1 = 0.0; double v0_y_dsp1 = 0.0;
    double vT_x_dsp1 = 0.0; double vT_y_dsp1 = 0.0;
    double v0_yaw_dsp1 = 0.0; double vT_yaw_dsp1 = 0.0;
    double v0_x_ssp  = 0.0; double v0_y_ssp = 0.0;
    double vT_x_ssp  = 0.0; double vT_y_ssp = 0.0;
    double v0_yaw_ssp  = 0.0; double vT_yaw_ssp = 0.0;
    double v0_x_dsp2 = 0.0; double v0_y_dsp2 = 0.0;
    double vT_x_dsp2 = 0.0; double vT_y_dsp2 = 0.0;
    double v0_yaw_dsp2 = 0.0; double vT_yaw_dsp2 = 0.0;

    double t_dsp1_ = t_dsp_(current_step_number);
    double t_dsp2_ = t_dsp_(current_step_number);
    double t_ssp = t_ssp_(current_step_number);
    double t_total = t_total_(current_step_number);    
    
    //TODO CoM Yaw implement
    if (current_step_number == 0)
    {
        // v0_x_dsp1 = com_support_init_yaw_(0);
        v0_x_dsp1 = (phase_indicator_(0)*lfoot_support_init_yaw_.translation()(0) + (1-phase_indicator_(0))*rfoot_support_init_yaw_.translation()(0))/2;
        vT_x_dsp1 = 0.0;
        // v0_y_dsp1 = com_support_init_yaw_(1);
        v0_y_dsp1 = (phase_indicator_(0)*lfoot_support_init_yaw_.translation()(1) + (1-phase_indicator_(0))*rfoot_support_init_yaw_.translation()(1))/2;
        vT_y_dsp1 = (2*phase_indicator_(current_step_number)-1)*zmp_offset;
        // v0_yaw_dsp1 =  pelv_support_euler_init_yaw_(2);
        v0_yaw_dsp1 = DyrosMath::rot2Euler(phase_indicator_(0)*lfoot_support_init_yaw_.linear() + (1-phase_indicator_(0))*rfoot_support_init_yaw_.linear())(2)/2;
        vT_yaw_dsp1 = v0_yaw_dsp1;

        v0_x_ssp = vT_x_dsp1;
        vT_x_ssp = v0_x_ssp;
        v0_y_ssp = vT_y_dsp1;
        vT_y_ssp = v0_y_ssp;
        v0_yaw_ssp =  vT_yaw_dsp1;
        vT_yaw_ssp = foot_step_support_frame_offset_(current_step_number - 0, 5) / 2.0;

        v0_x_dsp2 = vT_x_ssp;
        vT_x_dsp2 = (foot_step_support_frame_offset_(current_step_number - 0, 0)) / 2.0;
        v0_y_dsp2 = vT_y_ssp;
        vT_y_dsp2 = ((2*phase_indicator_(current_step_number - 0)-1)*zmp_offset + foot_step_support_frame_offset_(current_step_number - 0, 1)) / 2.0;
        v0_yaw_dsp2 = vT_yaw_ssp;
        vT_yaw_dsp2 = v0_yaw_dsp2;


    }
    else if (current_step_number == 1)
    { 
        v0_x_dsp1 = (foot_step_support_frame_offset_(current_step_number - 1, 0) ) / 2.0;
        vT_x_dsp1 =  foot_step_support_frame_offset_(current_step_number - 1, 0);
        v0_y_dsp1 = ((2*phase_indicator_(current_step_number - 1)-1)*zmp_offset + foot_step_support_frame_offset_(current_step_number - 1, 1)) / 2.0;
        vT_y_dsp1 =  foot_step_support_frame_offset_(current_step_number - 1, 1);
        v0_yaw_dsp1 = foot_step_support_frame_offset_(current_step_number - 1, 5) / 2.0;
        vT_yaw_dsp1 = v0_yaw_dsp1;

        v0_x_ssp = vT_x_dsp1;
        vT_x_ssp = v0_x_ssp;
        v0_y_ssp = vT_y_dsp1;
        vT_y_ssp = v0_y_ssp;
        v0_yaw_ssp = (foot_step_support_frame_offset_(current_step_number - 1, 5) )/ 2.0;
        vT_yaw_ssp = (foot_step_support_frame_offset_(current_step_number - 0, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;

        v0_x_dsp2 =  vT_x_ssp;
        vT_x_dsp2 = (foot_step_support_frame_offset_(current_step_number, 0) + foot_step_support_frame_offset_(current_step_number - 1, 0)) / 2.0;
        v0_y_dsp2 = vT_y_ssp;
        vT_y_dsp2 = (foot_step_support_frame_offset_(current_step_number, 1) + foot_step_support_frame_offset_(current_step_number - 1, 1)) / 2.0;
        v0_yaw_dsp2 = (foot_step_support_frame_offset_(current_step_number - 0, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;
        vT_yaw_dsp2 = (foot_step_support_frame_offset_(current_step_number - 0, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;

    }
    else
    {   
        v0_x_dsp1 = (foot_step_support_frame_offset_(current_step_number - 2, 0) + foot_step_support_frame_offset_(current_step_number - 1, 0)) / 2.0;
        vT_x_dsp1 =  foot_step_support_frame_offset_(current_step_number - 1, 0);
        v0_y_dsp1 = (foot_step_support_frame_offset_(current_step_number - 2, 1) + foot_step_support_frame_offset_(current_step_number - 1, 1)) / 2.0;
        vT_y_dsp1 =  foot_step_support_frame_offset_(current_step_number - 1, 1);
        v0_yaw_dsp1 = (foot_step_support_frame_offset_(current_step_number - 2, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;
        vT_yaw_dsp1 = (foot_step_support_frame_offset_(current_step_number - 2, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;

        v0_x_ssp = foot_step_support_frame_offset_(current_step_number - 1, 0);
        vT_x_ssp = foot_step_support_frame_offset_(current_step_number - 1, 0);
        v0_y_ssp = foot_step_support_frame_offset_(current_step_number - 1, 1);
        vT_y_ssp = foot_step_support_frame_offset_(current_step_number - 1, 1);
        v0_yaw_ssp =  (foot_step_support_frame_offset_(current_step_number - 2, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;
        vT_yaw_ssp = (foot_step_support_frame_offset_(current_step_number - 0, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;

        v0_x_dsp2 =  foot_step_support_frame_offset_(current_step_number - 1, 0);
        vT_x_dsp2 = (foot_step_support_frame_offset_(current_step_number - 1, 0) + foot_step_support_frame_offset_(current_step_number - 0, 0)) / 2.0;
        v0_y_dsp2 =  foot_step_support_frame_offset_(current_step_number - 1, 1);
        vT_y_dsp2 = (foot_step_support_frame_offset_(current_step_number - 1, 1) + foot_step_support_frame_offset_(current_step_number - 0, 1)) / 2.0;
        v0_yaw_dsp2 = (foot_step_support_frame_offset_(current_step_number - 0, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;
        vT_yaw_dsp2 = (foot_step_support_frame_offset_(current_step_number - 0, 5) + foot_step_support_frame_offset_(current_step_number - 1, 5))/ 2;
    }

    double lin_interpol = 0.0;
    for (int i = 0; i < t_total; i++)
    {
        if (i < t_dsp1_) 
        { 
            // lin_interpol = i / (t_dsp1_);
            // temp_px(i) = (1.0 - lin_interpol) * v0_x_dsp1 + lin_interpol * vT_x_dsp1;
            // temp_py(i) = (1.0 - lin_interpol) * v0_y_dsp1 + lin_interpol * vT_y_dsp1;
            
            temp_px(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, 0.0, t_dsp1_, v0_x_dsp1, vT_x_dsp1, 0.0, 0.0), min(v0_x_dsp1, vT_x_dsp1), max(v0_x_dsp1, vT_x_dsp1));
            temp_py(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, 0.0, t_dsp1_, v0_y_dsp1, vT_y_dsp1, 0.0, 0.0), min(v0_y_dsp1, vT_y_dsp1), max(v0_y_dsp1, vT_y_dsp1));
            temp_yaw(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, 0.0, t_dsp1_, v0_yaw_dsp1, vT_yaw_dsp1, 0.0, 0.0), min(v0_yaw_dsp1, vT_yaw_dsp1), max(v0_yaw_dsp1, vT_yaw_dsp1));
            temp_yawvel(i) = DyrosMath::cubicDot(i, 0.0, t_dsp1_, v0_yaw_dsp1, vT_yaw_dsp1, 0., 0.);
        }
        else if (i >= t_dsp1_ && i < t_dsp1_ + t_ssp)
        {
            // lin_interpol = (i - t_dsp1_) / t_ssp_;
            // temp_px(i) = (1.0 - lin_interpol) * v0_x_ssp + lin_interpol * vT_x_ssp;
            // temp_py(i) = (1.0 - lin_interpol) * v0_y_ssp + lin_interpol * vT_y_ssp;

            temp_px(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, t_dsp1_, t_dsp1_ + t_ssp, v0_x_ssp, vT_x_ssp, 0.0, 0.0), min(v0_x_ssp, vT_x_ssp), max(v0_x_ssp, vT_x_ssp));
            temp_py(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, t_dsp1_, t_dsp1_ + t_ssp, v0_y_ssp, vT_y_ssp, 0.0, 0.0), min(v0_y_ssp, vT_y_ssp), max(v0_y_ssp, vT_y_ssp));
            temp_yaw(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, t_dsp1_, t_dsp1_ + t_ssp, v0_yaw_ssp, vT_yaw_ssp, 0.0, 0.0), min(v0_yaw_ssp, vT_yaw_ssp), max(v0_yaw_ssp, vT_yaw_ssp));
            temp_yawvel(i) = DyrosMath::cubicDot(i, t_dsp1_, t_dsp1_+t_ssp, v0_yaw_ssp, vT_yaw_ssp, 0., 0.);
        }
        else
        {
            // lin_interpol = (i - t_dsp1_ - t_ssp_) / t_dsp2_;
            // temp_px(i) = (1.0 - lin_interpol) * v0_x_dsp2 + lin_interpol * vT_x_dsp2;
            // temp_py(i) = (1.0 - lin_interpol) * v0_y_dsp2 + lin_interpol * vT_y_dsp2;
            temp_px(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, t_dsp1_ + t_ssp, t_total, v0_x_dsp2, vT_x_dsp2, 0.0, 0.0), min(v0_x_dsp2, vT_x_dsp2), max(v0_x_dsp2, vT_x_dsp2));
            temp_py(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, t_dsp1_ + t_ssp, t_total, v0_y_dsp2, vT_y_dsp2, 0.0, 0.0), min(v0_y_dsp2, vT_y_dsp2), max(v0_y_dsp2, vT_y_dsp2));
            temp_yaw(i) = DyrosMath::minmax_cut(DyrosMath::cubic(i, t_dsp1_ + t_ssp, t_total, v0_yaw_dsp2, vT_yaw_dsp2, 0.0, 0.0), min(v0_yaw_dsp2, vT_yaw_dsp2), max(v0_yaw_dsp2, vT_yaw_dsp2));
            temp_yawvel(i) = DyrosMath::cubicDot(i, t_dsp1_ + t_ssp, t_total, v0_yaw_dsp2, vT_yaw_dsp2, 0., 0.);
        }
        // std::cout << current_step_number << " step's temp_py " << i << " : " << temp_py(i) << std::endl;
    }

}

void AvatarController::resetPreviewState(){
    x_preview_.setZero(); y_preview_.setZero(); 
    x_preview_(0) = com_support_init_yaw_(0);
    y_preview_(0) = com_support_init_yaw_(1);
    x_preview_(1) = com_support_init_dot_yaw_(0);
    y_preview_(1) = com_support_init_dot_yaw_(1);
    UX_preview_ = 0;
    UY_preview_ = 0;
    windupPreview();
}

void AvatarController::getComTrajectory()
{
    double dt_preview_ = 1.0 / hz_; // : sampling time of preview [s]
    double NL_preview  = 1.6 * hz_;      // : number of preview horizons

    if (is_preview_ctrl_init == true)
    {
        Gi_preview_.setZero();
        Gd_preview_.setZero();
        Gx_preview_.setZero();

        preview_Parameter(dt_preview_, NL_preview, Gi_preview_, Gd_preview_, Gx_preview_, A_preview_, B_preview_, C_preview_);
        
        resetPreviewState();

        is_preview_ctrl_init = false;

        std::cout << "PREVIEW PARAMETERS ARE SUCCESSFULLY INITIALIZED" << std::endl;
    }

    previewcontroller(dt_preview_, NL_preview, walking_tick, 
                      x_preview_, y_preview_, UX_preview_, UY_preview_,
                      Gi_preview_, Gd_preview_, Gx_preview_, 
                      A_preview_, B_preview_, C_preview_);

    com_desired_(0) = x_preview_(0);
    com_desired_(1) = y_preview_(0);
    com_desired_(2) = com_height_;
    com_desired_dot_(0) = x_preview_(1);
    com_desired_dot_(1) = y_preview_(1);
    com_desired_dot_(2) = 0.;

    if (!ideal_preview) windupPreview();

    
}

void AvatarController::preview_Parameter(double dt, int NL, Eigen::MatrixXd &Gi, Eigen::VectorXd &Gd, Eigen::MatrixXd &Gx, Eigen::MatrixXd &A, Eigen::VectorXd &B, Eigen::MatrixXd &C)
{
    A.resize(3, 3);
    A(0, 0) = 1.0;
    A(0, 1) = dt;
    A(0, 2) = dt * dt * 0.5;
    A(1, 0) = 0;
    A(1, 1) = 1.0;
    A(1, 2) = dt;
    A(2, 0) = 0;
    A(2, 1) = 0;
    A(2, 2) = 1;

    B.resize(3);
    B(0) = dt * dt * dt / 6;
    B(1) = dt * dt / 2;
    B(2) = dt;

    C.resize(1, 3);
    C(0, 0) = 1;
    C(0, 1) = 0;
    C(0, 2) = -com_height_ / GRAVITY;

    Eigen::MatrixXd A_bar;
    Eigen::VectorXd B_bar;

    B_bar.setZero(4);
    B_bar.segment(0, 1) = C * B;
    B_bar.segment(1, 3) = B;

    Eigen::Matrix1x4d B_bar_tran;
    B_bar_tran = B_bar.transpose();

    Eigen::MatrixXd I_bar;
    Eigen::MatrixXd F_bar;
    A_bar.setZero(4, 4);
    I_bar.setZero(4, 1);
    F_bar.setZero(4, 3);

    F_bar.block<1, 3>(0, 0) = C * A;
    F_bar.block<3, 3>(1, 0) = A;

    I_bar.setZero();
    I_bar(0, 0) = 1.0;

    A_bar.block<4, 1>(0, 0) = I_bar;
    A_bar.block<4, 3>(0, 1) = F_bar;

    Eigen::MatrixXd Qe;
    Qe.setZero(1, 1);
    Qe(0, 0) = 1.0;

    Eigen::MatrixXd R;
    R.setZero(1, 1);
    R(0, 0) = 0.000001;

    Eigen::MatrixXd Qx;
    Qx.setZero(3, 3);

    Eigen::MatrixXd Q_bar;
    Q_bar.setZero(3, 3);
    Q_bar(0, 0) = Qe(0, 0);

    Eigen::Matrix4d K; K.setZero();
    if (com_height_ == 0.7){
        K(0, 0) = 68.7921510770868;
        K(0, 1) = 2331.78394937073;
        K(0, 2) = 632.495145628673;
        K(0, 3) =2.57540412848618;
        K(1, 0) = 2331.78394937073;
        K(1, 1) = 81346.5405208585;
        K(1, 2) = 22074.2517082842;
        K(1, 3) = 92.2651171309122;
        K(2, 0) = 632.495145628673;
        K(2, 1) = 22074.2517082842;
        K(2, 2) = 5990.22394297225;
        K(2, 3) = 25.0754425025208;
        K(3, 0) = 2.57540412848618;
        K(3, 1) = 92.2651171309122;
        K(3, 2) = 25.0754425025208;
        K(3, 3) = 0.115349533788953;
    }

    else if (com_height_ == 0.728){
        K(0, 0) = 70.0813009391222;
        K(0, 1) = 2420.65372019101;
        K(0, 2) = 669.387885399937;
        K(0, 3) = 2.72127899669319;
        K(1, 0) = 2420.65372019101;
        K(1, 1) = 85969.0761591658;
        K(1, 2) = 23782.1340369413;
        K(1, 3) = 99.0855302680049;
        K(2, 0) = 669.387885399937;
        K(2, 1) = 23782.1340369413;
        K(2, 2) = 6579.12894080612;
        K(2, 3) = 27.4486400273521;
        K(3, 0) = 2.72127899669319;
        K(3, 1) = 99.0855302680049;
        K(3, 2) = 27.4486400273521;
        K(3, 3) = 0.125016968799296;
    }

    // // 0.65m
    else if (com_height_ == 0.65){
        K(0, 0) = 66.4281134896190;
        K(0, 1) = 2173.13307414793;
        K(0, 2) = 568.379709117415;
        K(0, 3) = 2.32202159217899;
        K(1, 0) = 2173.13307414793;
        K(1, 1) = 73309.6668377235;
        K(1, 2) = 19183.2680991541;
        K(1, 3) = 80.7128585695845;
        K(2, 0) = 568.379709117415;
        K(2, 1) = 19183.2680991541;
        K(2, 2) = 5019.91885890998;
        K(2, 3) = 21.1593587923669;
        K(3, 0) = 2.32202159217899;
        K(3, 1) = 80.7128585695845;
        K(3, 2) = 21.1593587923669;
        K(3, 3) = 0.0993494956670487;
    }


    // 0.68m
    else if (com_height_ == 0.68){
        K(0, 0) = 67.8563860285047;
        K(0, 1) = 2268.31636940822;
        K(0, 2) = 606.574465948491;
        K(0, 3) = 2.47294507780013;
        K(1, 0) = 2268.31636940822;
        K(1, 1) = 78097.9429536234;
        K(1, 2) = 20893.4283786941;
        K(1, 3) = 87.5477908301230;
        K(2, 0) = 606.574465948491;
        K(2, 1) = 20893.4283786941;
        K(2, 2) = 5589.73130292243;
        K(2, 3) = 23.4600661791894;
        K(3, 0) = 2.47294507780013;
        K(3, 1) = 87.5477908301230;
        K(3, 2) = 23.4600661791894;
        K(3, 3) = 0.108757490098902;
    }


    Eigen::MatrixXd Temp_mat;
    Eigen::MatrixXd Temp_mat_inv;
    Eigen::MatrixXd Ac_bar;
    Temp_mat.setZero(1, 1);
    Temp_mat_inv.setZero(1, 1);
    Ac_bar.setZero(4, 4);

    Temp_mat = R + B_bar_tran * K * B_bar;
    Temp_mat_inv = Temp_mat.inverse();

    Ac_bar = A_bar - B_bar * Temp_mat_inv * B_bar_tran * K * A_bar;

    Eigen::MatrixXd Ac_bar_tran(4, 4);
    Ac_bar_tran = Ac_bar.transpose();

    Gi.setZero(1, 1);
    Gx.setZero(1, 3);
    
    // 0.7m    
    if (com_height_ == 0.7){
        Gi(0, 0) = 562.367264784428; //Temp_mat_inv * B_bar_tran * K * I_bar ;
        //Gx = Temp_mat_inv * B_bar_tran * K * F_bar ;
        Gx(0, 0) = 38686.4538399671;
        Gx(0, 1) = 10800.0433241572;
        Gx(0, 2) = 128.255400226113;
    }

    // 0.728m
    else if (com_height_ == 0.728){
        Gi(0, 0) = 556.382091536873; //Temp_mat_inv * B_bar_tran * K * I_bar ;
        //Gx = Temp_mat_inv * B_bar_tran * K * F_bar ;
        Gx(0, 0) = 38991.9807941560;
        Gx(0, 1) = 11086.4028841705;
        Gx(0, 2) = 130.234568101964;
    }

    // 0.65m
    else if (com_height_ == 0.65){
        Gi(0, 0) = 573.462734668883; //Temp_mat_inv * B_bar_tran * K * I_bar ;
        //Gx = Temp_mat_inv * B_bar_tran * K * F_bar ;
        Gx(0, 0) = 38094.0476205746;
        Gx(0, 1) = 10274.4390649459;
        Gx(0, 2) = 124.583981245267;
    }

    // 0.68m
    else if (com_height_ == 0.68){
        Gi(0, 0) = 566.740708905623; //Temp_mat_inv * B_bar_tran * K * I_bar ;
        //Gx = Temp_mat_inv * B_bar_tran * K * F_bar ;
        Gx(0, 0) = 38456.9763214859;
        Gx(0, 1) = 10592.0336283141;
        Gx(0, 2) = 126.808547874660;
    }


    Eigen::MatrixXd X_bar;
    Eigen::Vector4d X_bar_col;
    X_bar.setZero(4, NL);
    X_bar_col.setZero();
    X_bar_col = -Ac_bar_tran * K * I_bar;

    for (int i = 0; i < NL; i++)
    {
        X_bar.block<4, 1>(0, i) = X_bar_col;
        X_bar_col = Ac_bar_tran * X_bar_col;
    }

    Gd.setZero(NL);
    Eigen::VectorXd Gd_col(1);
    Gd_col(0) = -Gi(0, 0);

    for (int i = 0; i < NL; i++)
    {
        Gd.segment(i, 1) = Gd_col;
        Gd_col = Temp_mat_inv * B_bar_tran * X_bar.col(i);
    }
}

void AvatarController::previewcontroller(double dt, int NL, int tick, 
                                         Eigen::Vector3d &x_k, Eigen::Vector3d &y_k, double &UX, double &UY,
                                         const Eigen::MatrixXd &Gi, const Eigen::VectorXd &Gd, const Eigen::MatrixXd &Gx, 
                                         const Eigen::MatrixXd &A,  const Eigen::VectorXd &B,  const Eigen::MatrixXd &C)
{

    Eigen::VectorXd px, py;
    px.setZero(1); px = C * x_k;
    py.setZero(1); py = C * y_k;
    EX_preview_ -= (px(0) - ref_zmp_(tick,0)) * Gi(0, 0);
    EY_preview_ -= (py(0) - ref_zmp_(tick,1)) * Gi(0, 0);
    double sum_Gd_px_ref = 0, sum_Gd_py_ref = 0;
    for (int i = 0; i < NL; i++)
    {
        sum_Gd_px_ref = sum_Gd_px_ref - Gd(i) * (ref_zmp_(tick + 1 + i,0));
        sum_Gd_py_ref = sum_Gd_py_ref - Gd(i) * (ref_zmp_(tick + 1 + i,1));
    }
    Eigen::VectorXd GX_X; GX_X.setZero(1);
    GX_X = -Gx * (x_k);
    Eigen::VectorXd GX_Y; GX_Y.setZero(1);
    GX_Y = -Gx * (y_k);

    UX = EX_preview_ + sum_Gd_px_ref + GX_X(0);
    UY = EY_preview_ + sum_Gd_py_ref + GX_Y(0);    

    x_k = A * x_k + B * UX;
    y_k = A * y_k + B * UY;    
}


void AvatarController::getFootTrajectory() 
{
    Eigen::Vector6d target_swing_foot; target_swing_foot.setZero();
    target_swing_foot = foot_step_support_frame_.row(0).transpose().segment(0,6);
    Eigen::Isometry3d &support_foot_traj           = (is_lfoot_support == true && is_rfoot_support == false) ? lfoot_trajectory_support_ : rfoot_trajectory_support_;
    Eigen::Vector3d &support_foot_traj_euler       = (is_lfoot_support == true && is_rfoot_support == false) ? lfoot_trajectory_euler_support_ : rfoot_trajectory_euler_support_;

    Eigen::Isometry3d &swing_foot_traj             = (is_lfoot_support == true && is_rfoot_support == false) ? rfoot_trajectory_support_ : lfoot_trajectory_support_;
    Eigen::Vector3d &swing_foot_traj_euler         = (is_lfoot_support == true && is_rfoot_support == false) ? rfoot_trajectory_euler_support_ : lfoot_trajectory_euler_support_;
    const Eigen::Isometry3d &swing_foot_init       = (is_lfoot_support == true && is_rfoot_support == false) ? rfoot_support_init_yaw_ : lfoot_support_init_yaw_;
    const Eigen::Vector3d &swing_foot_euler_init   = (is_lfoot_support == true && is_rfoot_support == false) ? rfoot_support_euler_init_yaw_ : lfoot_support_euler_init_yaw_;

    if (is_dsp1 == true)
    {
        support_foot_traj.translation().setZero();
        support_foot_traj_euler.setZero();

        swing_foot_traj.translation() = swing_foot_init.translation();
        // swing_foot_traj.translation()(2) = 0.0;
        swing_foot_traj_euler = swing_foot_euler_init;

        target_swing_state_stance_frame_.segment(7, 6) << 0., 0, 0, 0, 0, 0;
    }
    else if (is_ssp == true)
    {
        support_foot_traj.translation().setZero();
        support_foot_traj_euler.setZero();

        if (walking_tick < t_dsp_(0) + t_ssp_(0) / 2.0)
        {
            swing_foot_traj.translation()(2) = DyrosMath::cubic(walking_tick, 
                                                                t_dsp_(0),
                                                                t_dsp_(0) + t_ssp_(0) / 2.0, 
                                                                0.0, foot_height_(0), 
                                                                0.0, 0.0);
            target_swing_state_stance_frame_(9) = DyrosMath::cubicDot(walking_tick, 
                                                                t_dsp_(0),
                                                                t_dsp_(0) + t_ssp_(0) / 2.0, 
                                                                0.0, foot_height_(0), 
                                                                0.0, 0.0);

        }
        else
        {
            swing_foot_traj.translation()(2) = DyrosMath::cubic(walking_tick, 
                                                                t_dsp_(0) + t_ssp_(0) / 2.0,
                                                                t_dsp_(0) + t_ssp_(0), 
                                                                foot_height_(0), target_swing_foot(2), 
                                                                0.0, 0.0);
            target_swing_state_stance_frame_(9) = DyrosMath::cubicDot(walking_tick, 
                                                                t_dsp_(0) + t_ssp_(0) / 2.0,
                                                                t_dsp_(0) + t_ssp_(0), 
                                                                foot_height_(0), target_swing_foot(2), 
                                                                0.0, 0.0);                                                                
        }

        swing_foot_traj.translation().segment(0,2) = DyrosMath::cubicVector<2>(walking_tick, 
                                                                               t_dsp_(0),
                                                                               t_dsp_(0) + t_ssp_(0), 
                                                                               swing_foot_init.translation().segment(0,2), target_swing_foot.segment(0,2),
                                                                               Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
        target_swing_state_stance_frame_(7) = DyrosMath::cubicDot(walking_tick, 
                                                                               t_dsp_(0),
                                                                               t_dsp_(0) + t_ssp_(0), 
                                                                               swing_foot_init.translation()(0), target_swing_foot(0),
                                                                               0., 0.);
        target_swing_state_stance_frame_(8) = DyrosMath::cubicDot(walking_tick, 
                                                                               t_dsp_(0),
                                                                               t_dsp_(0) + t_ssp_(0), 
                                                                               swing_foot_init.translation()(1), target_swing_foot(1),
                                                                               0., 0.);

        swing_foot_traj_euler.setZero();
        swing_foot_traj_euler(2) = DyrosMath::cubic(walking_tick, 
                                                    t_dsp_(0), 
                                                    t_dsp_(0) + t_ssp_(0), 
                                                    swing_foot_euler_init(2), target_swing_foot(5), 
                                                    0.0, 0.0);
        target_swing_state_stance_frame_(12) =  DyrosMath::cubicDot(walking_tick, 
                                                    t_dsp_(0), 
                                                    t_dsp_(0) + t_ssp_(0), 
                                                    swing_foot_euler_init(2), target_swing_foot(5), 
                                                    0.0, 0.0);
    }
    else if (is_dsp2 == true)
    {
        support_foot_traj_euler.setZero();
        
        swing_foot_traj.translation() = target_swing_foot.segment(0,3);
        swing_foot_traj_euler = target_swing_foot.segment(3,3);
        target_swing_state_stance_frame_.segment(7, 6) << 0., 0, 0, 0, 0, 0;
    }

    swing_foot_traj.linear() = DyrosMath::Euler2rot(0., 0., swing_foot_traj_euler(2));
    support_foot_traj.linear() = DyrosMath::Euler2rot(support_foot_traj_euler(0), support_foot_traj_euler(1), support_foot_traj_euler(2));
    target_swing_state_stance_frame_.segment(0,3) = swing_foot_traj.translation();
    Eigen::Quaterniond swing_quat(swing_foot_traj.linear());
    target_swing_state_stance_frame_(3) = swing_quat.x();
    target_swing_state_stance_frame_(4) = swing_quat.y();
    target_swing_state_stance_frame_(5) = swing_quat.z();
    target_swing_state_stance_frame_(6) = swing_quat.w();

}

void AvatarController::computeIkControl(const Eigen::Isometry3d &float_trunk_transform, const Eigen::Isometry3d &float_lleg_transform, const Eigen::Isometry3d &float_rleg_transform, Eigen::Vector12d &q_des)
{
    Eigen::Vector3d R_r, R_D, L_r, L_D;

    L_D << 0.11, +0.1025, -0.1025;
    R_D << 0.11, -0.1025, -0.1025;

    L_r = float_lleg_transform.rotation().transpose() * (float_trunk_transform.translation() + float_trunk_transform.rotation() * L_D - float_lleg_transform.translation());
    R_r = float_rleg_transform.rotation().transpose() * (float_trunk_transform.translation() + float_trunk_transform.rotation() * R_D - float_rleg_transform.translation());

    double R_C = 0, L_C = 0, L_upper = 0.351, L_lower = 0.351, R_alpha = 0, L_alpha = 0;

    L_C = sqrt(pow(L_r(0), 2) + pow(L_r(1), 2) + pow(L_r(2), 2));
    R_C = sqrt(pow(R_r(0), 2) + pow(R_r(1), 2) + pow(R_r(2), 2));
     
    double knee_acos_var_L = 0;
    double knee_acos_var_R = 0;

    knee_acos_var_L = (pow(L_upper, 2) + pow(L_lower, 2) - pow(L_C, 2))/ (2 * L_upper * L_lower);
    knee_acos_var_R = (pow(L_upper, 2) + pow(L_lower, 2) - pow(R_C, 2))/ (2 * L_upper * L_lower);

    knee_acos_var_L = DyrosMath::minmax_cut(knee_acos_var_L, -0.99, + 0.99);
    knee_acos_var_R = DyrosMath::minmax_cut(knee_acos_var_R, -0.99, + 0.99);

    q_des(3) = (-acos(knee_acos_var_L) + M_PI);  
    q_des(9) = (-acos(knee_acos_var_R) + M_PI);
    q_des(5) = atan2(L_r(1), L_r(2));                                                                                  // Ankle roll
    q_des(11) = atan2(R_r(1), R_r(2));


    // L_alpha = asin(DyrosMath::minmax_cut(L_upper / L_C * sin(M_PI - q_des(3)), -0.99, 0.99) );

    // R_alpha = asin(DyrosMath::minmax_cut(L_upper / R_C * sin(M_PI - q_des(9)), -0.99, 0.99));

    L_alpha =asin( L_upper / L_C * sin(M_PI - q_des(3)));

    R_alpha = asin(L_upper / R_C * sin(M_PI - q_des(9)));
    
    q_des(4) = -atan2(L_r(0), sqrt(pow(L_r(1), 2) + pow(L_r(2), 2))) - L_alpha;
    q_des(10) = -atan2(R_r(0), sqrt(pow(R_r(1), 2) + pow(R_r(2), 2))) - R_alpha;

    Eigen::Matrix3d R_Knee_Ankle_Y_rot_mat, L_Knee_Ankle_Y_rot_mat;
    Eigen::Matrix3d R_Ankle_X_rot_mat, L_Ankle_X_rot_mat;
    Eigen::Matrix3d R_Hip_rot_mat, L_Hip_rot_mat;

    L_Knee_Ankle_Y_rot_mat = DyrosMath::rotateWithY(-q_des(3) - q_des(4));
    L_Ankle_X_rot_mat = DyrosMath::rotateWithX(-q_des(5));
    R_Knee_Ankle_Y_rot_mat = DyrosMath::rotateWithY(-q_des(9) - q_des(10));
    R_Ankle_X_rot_mat = DyrosMath::rotateWithX(-q_des(11));

    L_Hip_rot_mat.setZero();
    R_Hip_rot_mat.setZero();

    L_Hip_rot_mat = float_trunk_transform.rotation().transpose() * float_lleg_transform.rotation() * L_Ankle_X_rot_mat * L_Knee_Ankle_Y_rot_mat;
    R_Hip_rot_mat = float_trunk_transform.rotation().transpose() * float_rleg_transform.rotation() * R_Ankle_X_rot_mat * R_Knee_Ankle_Y_rot_mat;

    q_des(0) = atan2(-L_Hip_rot_mat(0, 1), L_Hip_rot_mat(1, 1));                                                       // Hip yaw
    q_des(1) = atan2(L_Hip_rot_mat(2, 1), -L_Hip_rot_mat(0, 1) * sin(q_des(0)) + L_Hip_rot_mat(1, 1) * cos(q_des(0))); // Hip roll
    q_des(2) = atan2(-L_Hip_rot_mat(2, 0), L_Hip_rot_mat(2, 2));                                                       // Hip pitch
    q_des(3) = q_des(3);                                                                                               // Knee pitch
    q_des(4) = q_des(4);                                                                                               // Ankle pitch

    q_des(6) = atan2(-R_Hip_rot_mat(0, 1), R_Hip_rot_mat(1, 1));
    q_des(7) = atan2(R_Hip_rot_mat(2, 1), -R_Hip_rot_mat(0, 1) * sin(q_des(6)) + R_Hip_rot_mat(1, 1) * cos(q_des(6)));
    q_des(8) = atan2(-R_Hip_rot_mat(2, 0), R_Hip_rot_mat(2, 2));
    q_des(9) = q_des(9);
    q_des(10) = q_des(10);
}



void AvatarController::getTargetState(){

    target_com_state_stance_frame_.segment(0, 3) = com_desired_;
    Eigen::Quaterniond com_quat(DyrosMath::rotateWithZ(ref_com_yaw_(walking_tick+1)));
    target_com_state_stance_frame_(3) = com_quat.x();
    target_com_state_stance_frame_(4) = com_quat.y();
    target_com_state_stance_frame_(5) = com_quat.z();
    target_com_state_stance_frame_(6) = com_quat.w();
    target_com_state_stance_frame_.segment(7, 3) = com_desired_dot_;
    target_com_state_stance_frame_(10) = 0.;
    target_com_state_stance_frame_(11) = 0.;
    target_com_state_stance_frame_(12) = ref_com_yawvel_(walking_tick+1);

    if (com_height_ == 0.68){
        target_com_state_float_frame_.translation() << DyrosMath::minmax_cut((rd_cc_.link_[Pelvis].rotm.transpose() * (rd_cc_.link_[Pelvis].xpos-rd_cc_.link_[COM_id].xpos))(0), -0.15, 0.),
        DyrosMath::minmax_cut((rd_cc_.link_[Pelvis].rotm.transpose() * (rd_cc_.link_[Pelvis].xpos-rd_cc_.link_[COM_id].xpos))(1), -0.02, 0.02), 
        DyrosMath::minmax_cut((rd_cc_.link_[Pelvis].rotm.transpose() * (rd_cc_.link_[Pelvis].xpos-rd_cc_.link_[COM_id].xpos))(2), 0., 0.04);

    }
    else if (com_height_ == 0.728){
        target_com_state_float_frame_.translation() << DyrosMath::minmax_cut((rd_cc_.link_[Pelvis].rotm.transpose() * (rd_cc_.link_[Pelvis].xpos-rd_cc_.link_[COM_id].xpos))(0), -0.15, 0.),
        DyrosMath::minmax_cut((rd_cc_.link_[Pelvis].rotm.transpose() * (rd_cc_.link_[Pelvis].xpos-rd_cc_.link_[COM_id].xpos))(1), -0.04, 0.04), 
        DyrosMath::minmax_cut((rd_cc_.link_[Pelvis].rotm.transpose() * (rd_cc_.link_[Pelvis].xpos-rd_cc_.link_[COM_id].xpos))(2), 0., 0.04);
    }
    else{
        std::cout << "WRONG COM HEIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
        exit(0);
    }
    target_com_state_float_frame_.linear() = Eigen::Matrix3d::Identity();

    Eigen::Vector4d swingq = target_swing_state_stance_frame_.segment(3,4);
    Eigen::Vector4d comq = target_com_state_stance_frame_.segment(3,4);
    Eigen::Quaterniond target_swing_state_stance_frame_quat_(swingq(3), swingq(0), swingq(1), swingq(2));
    Eigen::Quaterniond target_com_state_stance_frame_quat_(comq(3), comq(0), comq(1), comq(2));

    target_lfoot_state_float_frame_.translation() = phase_indicator_(0)*target_com_state_stance_frame_quat_.toRotationMatrix().transpose() * (target_swing_state_stance_frame_.segment(0,3) - target_com_state_stance_frame_.segment(0,3)) + (1-phase_indicator_(0)) * target_com_state_stance_frame_quat_.toRotationMatrix().transpose()*(-target_com_state_stance_frame_.segment(0,3));
    target_lfoot_state_float_frame_.linear() = phase_indicator_(0)*target_com_state_stance_frame_quat_.toRotationMatrix().transpose()*target_swing_state_stance_frame_quat_.toRotationMatrix() + (1-phase_indicator_(0))*target_com_state_stance_frame_quat_.toRotationMatrix().transpose();
    target_rfoot_state_float_frame_.translation() = (1-phase_indicator_(0))*target_com_state_stance_frame_quat_.toRotationMatrix().transpose() * (target_swing_state_stance_frame_.segment(0,3) - target_com_state_stance_frame_.segment(0,3)) + phase_indicator_(0) * target_com_state_stance_frame_quat_.toRotationMatrix().transpose()*(-target_com_state_stance_frame_.segment(0,3));
    target_rfoot_state_float_frame_.linear() = (1-phase_indicator_(0))*target_com_state_stance_frame_quat_.toRotationMatrix().transpose()*target_swing_state_stance_frame_quat_.toRotationMatrix() + phase_indicator_(0)*target_com_state_stance_frame_quat_.toRotationMatrix().transpose();

    computeIkControl(target_com_state_float_frame_, target_lfoot_state_float_frame_, target_rfoot_state_float_frame_, q_leg_desired_);
}


void AvatarController::windupPreview(){
    EX_preview_ = 0.;
    EY_preview_ = 0.;
}

void AvatarController::updateNextStepTime()
{       
    // std::cout << "walking time : " << (walking_tick) / hz_ <<  ", Value : " << value_ << std::endl;
    walking_tick++;
}

void AvatarController::walkingStateMachine()
{
    if (foot_step_(0, 6) == 1) 
    {
        is_lfoot_support = true;
        is_rfoot_support = false;
    }
    else if (foot_step_(0, 6) == 0) 
    {
        is_lfoot_support = false;
        is_rfoot_support = true;
    }

    if (walking_tick < t_dsp_(0))
    {
        is_dsp1 = true;
        is_ssp  = false;
        is_dsp2 = false;
    }
    else if (walking_tick >= t_dsp_(0) && walking_tick < t_total_(0) - t_dsp_(0))
    {
        is_dsp1 = false;
        is_ssp  = true;
        is_dsp2 = false;
    }
    else
    {
        is_dsp1 = false;
        is_ssp  = false;
        is_dsp2 = true;
    } 
}


void AvatarController::computePlanner()
{
}

void AvatarController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}