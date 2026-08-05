load('Drug_id.mat');
load('data2alex.mat')
Subjects = [1:20];
sessions =1:6;
% some subjects have an extra repeat 7th block
blocks =1:7;

Trial_number=nan(20,6,7);
n_hist_trials = 0;
non_rep_trans = [];
rep_trans = [];
All_Data_s = All_Data_s(All_Drug==2,:); % placebo only
All_Data_s = All_Data_s(~isnan(All_Data_s(:,1)),:);
for si=1:20
    for ses=1:6
        for bl=1:7
          id_tr=find(All_Data_s(:,25)==bl & All_Data_s(:,24)==si & All_Data_s(:,26)==ses);
          Trial_number(si,ses,bl)=numel(id_tr);
          if numel(id_tr) > 120
            disp("Too many trials: "+numel(id_tr))
            continue
          end
          if numel(id_tr) > 2
            n_hist_trials = n_hist_trials + 1;
            % ground truth
            lr = sign(All_Data_s(id_tr,23));
            %lr = lr(lr~=0);
            trans = diff(lr)';
            %trans = abs(trans/2);
            if All_Data_s(id_tr(1),27) == 0.5
              non_rep_trans = [non_rep_trans, trans];
            else
              disp(id_tr(1))
              rep_trans = [rep_trans, trans];
            end
          end
        end
    end
end
disp("Total number of trials: "+sum(Trial_number(:)))
disp("Total number of block trials: "+n_hist_trials)
disp("Transition probability in random trials: "+mean(non_rep_trans~=0))
disp("Transition probability in history trials: "+mean(rep_trans~=0))