DROP TABLE IF EXISTS field_operations CASCADE;
DROP TABLE IF EXISTS environmental_reports CASCADE;
DROP TABLE IF EXISTS contracts CASCADE;
DROP TABLE IF EXISTS contractors CASCADE;
DROP TABLE IF EXISTS geological_surveys CASCADE;
DROP TABLE IF EXISTS reservoirs CASCADE;
DROP TABLE IF EXISTS inspections CASCADE;
DROP TABLE IF EXISTS maintenance_records CASCADE;
DROP TABLE IF EXISTS drilling_operations CASCADE;
DROP TABLE IF EXISTS rigs CASCADE;
DROP TABLE IF EXISTS facilities CASCADE;
DROP TABLE IF EXISTS production_charts CASCADE;
DROP TABLE IF EXISTS difa CASCADE;
DROP TABLE IF EXISTS outstanding_passeds CASCADE;
DROP TABLE IF EXISTS outstanding_exceptions CASCADE;
DROP TABLE IF EXISTS well_extra_info CASCADE;
DROP TABLE IF EXISTS well_test_metrics CASCADE;
DROP TABLE IF EXISTS well_tests CASCADE;
DROP TABLE IF EXISTS well_production CASCADE;
DROP TABLE IF EXISTS well_status CASCADE;
DROP TABLE IF EXISTS well_locations CASCADE;
DROP TABLE IF EXISTS wells CASCADE;
DROP TABLE IF EXISTS workflow_steps CASCADE;
DROP TABLE IF EXISTS passed_signals CASCADE;
DROP TABLE IF EXISTS exception_signals CASCADE;
DROP TABLE IF EXISTS workflows CASCADE;
DROP TABLE IF EXISTS actions CASCADE;
DROP TABLE IF EXISTS exceptions CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS roles CASCADE;
DROP TABLE IF EXISTS dca CASCADE;
DROP TABLE IF EXISTS wellbore_components CASCADE;

CREATE TABLE roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(255) NOT NULL,
    role_desc TEXT
);

CREATE TABLE users (
    user_id VARCHAR(255) PRIMARY KEY
);

CREATE TABLE exceptions (
    except_code VARCHAR(255) PRIMARY KEY,
    except_name TEXT,
    except_desc TEXT,
    except_kind TEXT,
    except_type TEXT,
    priority INT,
    total_exception INT
);

CREATE TABLE actions (
    action_id VARCHAR(255) PRIMARY KEY,
    action_code TEXT,
    action_name TEXT,
    action_desc TEXT,
    action_type TEXT,
    total_action INT,
    type_data TEXT
);

CREATE TABLE workflows (
    workflow_id VARCHAR(255) PRIMARY KEY
);

CREATE TABLE exception_signals (
    exception_signal_id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES USERS(user_id),
    role_id INT REFERENCES ROLES(role_id),
    except_code VARCHAR(255) REFERENCES EXCEPTIONS(except_code)
);

CREATE TABLE passed_signals (
    passed_signal_id SERIAL PRIMARY KEY,
    role_id INT REFERENCES ROLES(role_id),
    action_id VARCHAR(255) REFERENCES ACTIONS(action_id),
    workflow_id VARCHAR(255) REFERENCES WORKFLOWS(workflow_id)
);

CREATE TABLE workflow_steps (
    workflow_step_id SERIAL PRIMARY KEY,
    action_id VARCHAR(255) REFERENCES ACTIONS(action_id),
    step_number INT,
    role_id INT REFERENCES ROLES(role_id),
    step_desc TEXT,
    action_taken TEXT,
    workflow_status TEXT,
    remark TEXT,
    inputed_by VARCHAR(255),
    inputed_date TIMESTAMP,
    workflow_id VARCHAR(255) REFERENCES WORKFLOWS(workflow_id),
    bf_action_taken TEXT,
    filename TEXT,
    files TEXT,
    attachment_id TEXT
);

CREATE TABLE wells (
    well_id SERIAL PRIMARY KEY,
    uwi VARCHAR(255),
    string_code VARCHAR(255),
    string_name VARCHAR(255),
    grid_name VARCHAR(255),
    alt_name VARCHAR(255)
);

CREATE TABLE well_locations (
    location_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    sub_production_unit_code VARCHAR(255),
    field VARCHAR(255),
    area VARCHAR(255),
    area_code VARCHAR(255),
    arse VARCHAR(255),
    sub_area_code VARCHAR(255),
    fcty1_code VARCHAR(255),
    fcty2_code VARCHAR(255),
    region VARCHAR(255),
    sub_region VARCHAR(255),
    asset_area VARCHAR(255)
);

CREATE TABLE well_status (
    status_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    status_start_date TIMESTAMP,
    pis_date TIMESTAMP,
    drill_type VARCHAR(255),
    prod_string_type VARCHAR(255),
    status_type VARCHAR(255),
    status VARCHAR(255),
    awt_tier INT,
    global_tier INT,
    sand VARCHAR(255),
    completion_type VARCHAR(255),
    commingle_status VARCHAR(255),
    wt_date TIMESTAMP,
    lowis_alarm VARCHAR(255),
    lowis_scandate TIMESTAMP,
    dyno_test_date TIMESTAMP
);

CREATE TABLE well_production (
    production_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    avg_alloc_bfpd3_mo INT,
    avg_alloc_bopd3_mo INT,
    avg_alloc_bwpd3_mo INT,
    avg_alloc_gas3_mo INT,
    avg_theor_bfpd3_mo INT,
    avg_theor_bopd3_mo INT,
    avg_theor_bwpd3_mo INT,
    avg_theor_gas3_mo INT,
    avg_wt_bfpd3_mo DECIMAL,
    avg_wt_bopd3_mo DECIMAL,
    avg_wt_bwpd3_mo DECIMAL,
    avg_wt_wc3_mo DECIMAL,
    avg_wt_gas3_mo VARCHAR(255),
    avg5_wt_bf_bopd DECIMAL,
    avg5_wt_bf_bfpd DECIMAL,
    avg5_wt_bf_bwpd DECIMAL,
    alloc_bfpd INT,
    alloc_bopd INT,
    alloc_bwpd INT,
    alloc_gas VARCHAR(255),
    theor_bfpd INT,
    theor_bopd INT,
    theor_bwpd INT,
    theor_gas VARCHAR(255),
    wt_bfpd DECIMAL,
    wt_bopd DECIMAL,
    wt_bwpd DECIMAL,
    wt_wc DECIMAL,
    gor INT
);

CREATE TABLE well_tests (
    test_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    pump_fill VARCHAR(255),
    pump_slip VARCHAR(255),
    gross_disp VARCHAR(255),
    net_disp VARCHAR(255),
    pump_speed VARCHAR(255),
    pump_stroke VARCHAR(255),
    dyno_wht VARCHAR(255),
    tubing_press_up VARCHAR(255),
    tubing_press_down VARCHAR(255),
    csg_press_dn VARCHAR(255),
    install_date TIMESTAMP,
    pump_type VARCHAR(255),
    pump_info VARCHAR(255),
    pump_size VARCHAR(255),
    plug_back_depth VARCHAR(255),
    pump_depth VARCHAR(255),
    tubing_size VARCHAR(255),
    prdcsg_od VARCHAR(255),
    prdcsg_bott VARCHAR(255),
    scrliner_od VARCHAR(255),
    scrliner_top VARCHAR(255),
    scrliner_bott VARCHAR(255),
    down_date TIMESTAMP,
    hrs_since_off VARCHAR(255),
    days_since_off VARCHAR(255),
    bopd_defered VARCHAR(255),
    total_down_oil VARCHAR(255),
    sor_ss VARCHAR(255),
    proact_or_react VARCHAR(255)
);

CREATE TABLE well_test_metrics (
    metric_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    avg_pfill3_mo VARCHAR(255),
    avg_pslip3_mo VARCHAR(255),
    avg_gross_disp3_mo VARCHAR(255),
    avg_net_disp3_mo VARCHAR(255),
    avg_pump_speed3_mo VARCHAR(255),
    avg_pstroke3_mo VARCHAR(255),
    avg_wht3_mo VARCHAR(255),
    avg_tbgp_up3_mo VARCHAR(255),
    avg_tbgp_down3_mo VARCHAR(255),
    avg_csgp_dn3_mo VARCHAR(255)
);

CREATE TABLE well_extra_info (
    extra_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    wt_train VARCHAR(255),
    commingle_list VARCHAR(255),
    manifold VARCHAR(255),
    new_manifold VARCHAR(255)
);

CREATE TABLE outstanding_exceptions (
    out_exception_id SERIAL PRIMARY KEY,
    signal_code VARCHAR(255),
    well_id INT REFERENCES WELLS(well_id),
    string_code VARCHAR(255),
    string_name VARCHAR(255),
    status_well VARCHAR(255),
    current_well_status VARCHAR(255),
    signal_remark VARCHAR(255),
    total_exception VARCHAR(255),
    list_exception VARCHAR(255),
    last_wt_date TIMESTAMP,
    last_bopd INT,
    delta_bopd INT,
    last_bfpd INT,
    delta_bfpd INT,
    type_data VARCHAR(255),
    except_code VARCHAR(255) REFERENCES EXCEPTIONS(except_code),
    pump_type VARCHAR(255),
    alt_name VARCHAR(255),
    grid_name VARCHAR(255),
    wt_train VARCHAR(255),
    new_manifold VARCHAR(255),
    commingle_list VARCHAR(255),
    fluid_above_pump VARCHAR(255),
    prod_string_type VARCHAR(255)
);

CREATE TABLE outstanding_passeds (
    out_passed_id SERIAL PRIMARY KEY,
    action_id VARCHAR(255) REFERENCES ACTIONS(action_id),
    workflow_id VARCHAR(255) REFERENCES WORKFLOWS(workflow_id),
    ids VARCHAR(255),
    action_status VARCHAR(255),
    action_code VARCHAR(255),
    action_desc VARCHAR(255),
    well_id INT REFERENCES WELLS(well_id),
    string_code VARCHAR(255),
    string_name VARCHAR(255),
    status_well VARCHAR(255),
    action_flow_step VARCHAR(255),
    percentage VARCHAR(255),
    finding VARCHAR(255),
    job_plan VARCHAR(255),
    remark VARCHAR(255),
    est_bopd VARCHAR(255),
    dpi VARCHAR(255),
    passed_by VARCHAR(255),
    passed_date TIMESTAMP,
    action_originator VARCHAR(255),
    action_date TIMESTAMP,
    workflow_step INT,
    total_workflow_step INT,
    alt_name VARCHAR(255),
    grid_name VARCHAR(255),
    role_id INT REFERENCES ROLES(role_id),
    wt_train VARCHAR(255),
    new_manifold VARCHAR(255),
    ptm_region VARCHAR(255),
    zona VARCHAR(255)
);

CREATE TABLE difa (
    difa_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    string_id VARCHAR(255),
    install_date TIMESTAMP,
    pump_sno VARCHAR(255),
    pull_date TIMESTAMP,
    inpection_date TIMESTAMP,
    dismantled_date TIMESTAMP,
    received_date TIMESTAMP,
    out_diam VARCHAR(255),
    pump_brand VARCHAR(255),
    sand VARCHAR(255),
    mud VARCHAR(255),
    oil VARCHAR(255),
    scale VARCHAR(255),
    reason_of_failure VARCHAR(255),
    shop_finding1 VARCHAR(255),
    shop_finding2 VARCHAR(255),
    shop_finding3 VARCHAR(255),
    shop_finding4 VARCHAR(255),
    shop_finding5 VARCHAR(255),
    recommendation VARCHAR(255),
    remark VARCHAR(255)
);

CREATE TABLE production_charts (
    chart_id SERIAL PRIMARY KEY,
    string_code VARCHAR(255),
    test_date TIMESTAMP,
    int_test_date TIMESTAMP,
    pump_fill VARCHAR(255),
    pump_slip VARCHAR(255),
    gross_disp VARCHAR(255),
    net_disp VARCHAR(255),
    spm VARCHAR(255),
    sl VARCHAR(255),
    tubing_press VARCHAR(255),
    csg_press_up VARCHAR(255),
    wht VARCHAR(255),
    tst_fluid VARCHAR(255),
    tst_oil VARCHAR(255),
    tst_water VARCHAR(255),
    wc VARCHAR(255),
    tst_oil_no_acc VARCHAR(255),
    alloc_fluid VARCHAR(255),
    alloc_oil VARCHAR(255),
    theor_fluid VARCHAR(255),
    theor_oil VARCHAR(255),
    job_code VARCHAR(255),
    yes_trun VARCHAR(255),
    updated_date TIMESTAMP,
    inferred_bfpd VARCHAR(255),
    inferred_bwpd VARCHAR(255),
    inferred_bopd VARCHAR(255),
    freq VARCHAR(255),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    job_status VARCHAR(255),
    remark VARCHAR(255),
    job_detail_description VARCHAR(255),
    job_category VARCHAR(255),
    rig_code VARCHAR(255)
);

CREATE TABLE facilities (
    facility_id SERIAL PRIMARY KEY,
    facility_name VARCHAR(255),
    facility_type VARCHAR(255),
    location_description VARCHAR(255),
    installed_date TIMESTAMP,
    capacity VARCHAR(255),
    operating_status VARCHAR(255)
);

CREATE TABLE rigs (
    rig_id SERIAL PRIMARY KEY,
    rig_name VARCHAR(255),
    rig_type VARCHAR(255),
    manufacturer VARCHAR(255),
    commissioning_date TIMESTAMP,
    decommission_date TIMESTAMP
);

CREATE TABLE drilling_operations (
    drilling_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    rig_id INT REFERENCES RIGS(rig_id),
    drilling_start_date TIMESTAMP,
    drilling_end_date TIMESTAMP,
    drill_operator VARCHAR(255),
    remarks VARCHAR(255)
);

CREATE TABLE maintenance_records (
    maintenance_id SERIAL PRIMARY KEY,
    target_type VARCHAR(255),
    target_id INT,
    maintenance_date TIMESTAMP,
    maintenance_type VARCHAR(255),
    performed_by VARCHAR(255),
    description VARCHAR(255)
);

CREATE TABLE inspections (
    inspection_id SERIAL PRIMARY KEY,
    target_type VARCHAR(255),
    target_id INT,
    inspection_date TIMESTAMP,
    inspector VARCHAR(255),
    inspection_result VARCHAR(255),
    remarks VARCHAR(255)
);

CREATE TABLE reservoirs (
    reservoir_id SERIAL PRIMARY KEY,
    reservoir_name VARCHAR(255),
    associated_field VARCHAR(255),
    depth VARCHAR(255),
    porosity VARCHAR(255),
    permeability VARCHAR(255),
    well_id INT REFERENCES WELLS(well_id)
);

CREATE TABLE geological_surveys (
    survey_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    survey_date TIMESTAMP,
    geologist VARCHAR(255),
    summary VARCHAR(255),
    detailed_report VARCHAR(255)
);

CREATE TABLE contractors (
    contractor_id SERIAL PRIMARY KEY,
    contractor_name VARCHAR(255),
    service_offered VARCHAR(255),
    contact_info VARCHAR(255)
);

CREATE TABLE contracts (
    contract_id SERIAL PRIMARY KEY,
    contractor_id INT REFERENCES CONTRACTORS(contractor_id),
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    service_description VARCHAR(255),
    contract_value DECIMAL
);

CREATE TABLE environmental_reports (
    report_id SERIAL PRIMARY KEY,
    target_type VARCHAR(255),
    target_id INT,
    report_date TIMESTAMP,
    reporter VARCHAR(255),
    findings VARCHAR(255),
    compliance_status VARCHAR(255),
    remarks VARCHAR(255)
);

CREATE TABLE field_operations (
    operation_id SERIAL PRIMARY KEY,
    well_id INT REFERENCES WELLS(well_id),
    operation_date TIMESTAMP,
    operation_type VARCHAR(255),
    shift_leader VARCHAR(255),
    summary VARCHAR(255),
    remarks VARCHAR(255)
);

CREATE TABLE dca (
    dca_id SERIAL PRIMARY KEY,
    string_code VARCHAR(255),
    test_date TIMESTAMP,
    tstfluid DECIMAL,
    tstoil DECIMAL,
    job_code VARCHAR(255)
)