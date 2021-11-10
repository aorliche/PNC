%% write_wrat_csv.m

% Write WRAT scores from cnb_data.mat to CSV files
% keyed on subject ID

% data is found cnb_data struct
% If first column starts with ID excel will mistake .csv for SYLK file

fname = 'wrat.csv';
fid = fopen(fname, 'w');
fprintf(fid, 'PNCID,Valid,Raw,Std\n');

for i=1:length(cnb_data)
    id = cnb_data(i).meta.id;
    valid = cnb_data(i).res_wrat.wrat_valid;
    raw = cnb_data(i).res_wrat.wrat_cr_raw;
    std = cnb_data(i).res_wrat.wrat_cr_std;
    
    fprintf(fid, "%s,%s,%d,%d\n",id,valid,raw,std);
end

fclose(fid);