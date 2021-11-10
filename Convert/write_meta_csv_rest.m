%% write_meta_csv_rest.m

% Write meta infromation for 878 rest_fmri_power264.mat patients to a 
% CSV file

% Data is located in rest_fmri_power264(i).meta struct, but this is not
% convenient for reading from python

% for i=1:878
%     id = rest_fmri_power264(i).meta.id;
%     if id == round(id)
%         fprintf('got one: %d\n', i);
%     end
% end
% 
% error('bad')

fname = 'rest_fmri_power264_meta.csv';
fid = fopen(fname, 'w');
fprintf(fid, 'PythonID,ID,AgeInMonths,Gender,Ethnicity,AgeGroupID,AgeGroupEdge1,AgeGroupEdge2\n');

for i=1:878
    py_id = i-1;
    id = rest_fmri_power264(i).meta.id;
    age_in_month = rest_fmri_power264(i).meta.age_in_month;
    gender = rest_fmri_power264(i).meta.gender;
    ethnicity = rest_fmri_power264(i).meta.ethnicity;
    age_grp_id = rest_fmri_power264(i).meta.age_grp_id;
    age_grp_edges = rest_fmri_power264(i).meta.age_grp_edges;
    
    fprintf(fid, '%d,%ld,%d,%s,%s,%d,%d,%d\n', py_id,id,age_in_month,gender,ethnicity,age_grp_id,age_grp_edges);
end

fclose(fid);