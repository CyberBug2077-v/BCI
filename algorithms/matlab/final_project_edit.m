clc;clear;close all;
ctest=zeros(1,7);
ctrain=zeros(1,7);

fs=200;                                % Sampling Frequency
ts=1/fs;                               % Sampling period
T=0.5;                                 % Epoch length for feature extracting  &&&
p=10;                                  % Model order                           &&&
%%% load signals

Sig_1=load('C:\Users\Hossein\Desktop\hekmat result\hekmat 2\Hekmat_second.mat');
Sig_2=load('C:\Users\Hossein\Desktop\hekmat result\sadeeghi\sadeghi_second.mat');
Sig_3=load('C:\Users\Hossein\Desktop\hekmat result\nafisieh\nafiseh_second.mat');
Sig_4_1=load('C:\Users\Hossein\Desktop\hekmat result\hekmat1\1.mat');
Sig_4_2=load('C:\Users\Hossein\Desktop\hekmat result\hekmat1\2.mat');
Sig_4_3=load('C:\Users\Hossein\Desktop\hekmat result\hekmat1\3.mat');

Sig_5_1=load('C:\Users\Hossein\Desktop\hekmat result\anjidani1\angidani1_1.mat');
Sig_5_2=load('C:\Users\Hossein\Desktop\hekmat result\anjidani1\angidani1_2.mat');
Sig_6_1=load('C:\Users\Hossein\Desktop\hekmat result\anjidani2\anjidani2_1.mat');
Sig_6_2=load('C:\Users\Hossein\Desktop\hekmat result\anjidani2\anjidani2_2.mat');
Sig_7_1=load('C:\Users\Hossein\Desktop\hekmat result\hadizadeh1\hadizadeh1_1.mat');
Sig_7_2=load('C:\Users\Hossein\Desktop\hekmat result\hadizadeh1\hadizadeh1_2.mat');
Sig_8_1=load('C:\Users\Hossein\Desktop\hekmat result\molavi1\molavi1_1.mat');
Sig_8_2=load('C:\Users\Hossein\Desktop\hekmat result\molavi1\molavi1_2.mat');
Sig_9_1=load('C:\Users\Hossein\Desktop\hekmat result\molavi2\molavi2_1.mat');
Sig_9_2=load('C:\Users\Hossein\Desktop\hekmat result\molavi2\molavi2_2.mat');
Sig_10_1=load('C:\Users\Hossein\Desktop\hekmat result\pajakh2\pajakh2_1.mat');
% Sig_10_2=load('C:\Users\Hossein\Desktop\hekmat result\pajakh2\pajakh2_2.mat');
% Sig_11_1=load('C:\Users\Hossein\Desktop\hekmat result\sadeghi1\sadeghi1_1.mat');
% Sig_11_2=load('C:\Users\Hossein\Desktop\hekmat result\sadeghi1\sadeghi1_2.mat');
Sig_12_1=load('C:\Users\Hossein\Desktop\hekmat result\saidi1\saidi1_1.mat');
Sig_12_2=load('C:\Users\Hossein\Desktop\hekmat result\saidi1\saidi1_2.mat');



% Sig_3=load('C:\Users\Hossein\Desktop\hekmat result\pourghani\pourghani_five.mat');

Sig_1=Sig_1.X;
Sig_2=Sig_2.X;
Sig_3=Sig_3.X2';
Sig_4_1=Sig_4_1.X1';
Sig_4_2=Sig_4_2.X2';
Sig_4_3=Sig_4_3.X3';

Sig_5_1=Sig_5_1.X1';
Sig_5_2=Sig_5_2.X2';
Sig_6_1=Sig_6_1.X1';
Sig_6_2=Sig_6_2.X2';
Sig_7_1=Sig_7_1.X1';
Sig_7_2=Sig_7_2.X2';
Sig_8_1=Sig_8_1.X1';
Sig_8_2=Sig_8_2.X2';
Sig_9_1=Sig_9_1.X1';
Sig_9_2=Sig_9_2.X2';
Sig_10_1=Sig_10_1.X1';
% Sig_10_2=Sig_10_2.X2';
% Sig_11_1=Sig_11_1.X2';
% Sig_11_2=Sig_11_2.X1';
Sig_12_1=Sig_12_1.X1';
Sig_12_2=Sig_12_2.saidi1_2';

%%%%%%%%%%%%%%%%%%%%% All of the signals

Sig1_1  = Sig_1(1,17977:end);
Sig1_2  = Sig_1(2,17968:end);
Sig1_3  = Sig_1(3,17968:end);
Sig2_1  = Sig_2(1,53:end);
Sig2_2  = Sig_2(2,32:end);
Sig2_3  = Sig_2(3,13:end);
Sig3_1  = Sig_3(1,23980:end);
Sig3_2  = Sig_3(1,23978:end);
Sig_4_1  = Sig_4_1(1,29985:end);
Sig_4_2  = Sig_4_2(1,29985:end);
Sig_4_3  = Sig_4_3(1,29985:end);
Sig_5_1 =[zeros(1,40) Sig_5_1];
Sig_5_2 =[zeros(1,64) Sig_5_2];
Sig_6_1 =[zeros(1,42) Sig_6_1];
Sig_6_2 =[zeros(1,85) Sig_6_2];
Sig_7_1 =[zeros(1,17) Sig_7_1];
Sig_7_2 =[zeros(1,22) Sig_7_2];
Sig_8_1 =[zeros(1,34) Sig_8_1];
Sig_8_2 =[zeros(1,35) Sig_8_2];
Sig_9_1 =[zeros(1,30) Sig_9_1];
Sig_9_2 =[zeros(1,25) Sig_9_2];
Sig_10_1=[zeros(1,12) Sig_10_1];
Sig_12_1=[zeros(1,35) Sig_12_1];
Sig_12_2=[zeros(1,2) Sig_12_2];

yyy=0;
% L1=length(S2_1);L2=length(S2_2);L3=length(S2_3);L4=length(S2_4);L5=length(S2_5);

% timing1=[S2_1,zeros(5,L2-L1);S2_2;S2_3,zeros(5,L2-L3);S2_4,zeros(5,L2-L4);S2_5 zeros(5,L2-L5)];

%%%%%%%%%%      load seconds of spindles

t1=load('C:\Users\Hossein\Desktop\hekmat result\hekmat 2\time_tot_Sp_hekmat2.mat');
t2=load('C:\Users\Hossein\Desktop\hekmat result\sadeeghi\time_tot_Sp_sadeghi.mat');
t3=load('C:\Users\Hossein\Desktop\hekmat result\nafisieh\time_tot_Sp_saidi.mat');
t4=load('C:\Users\Hossein\Desktop\hekmat result\hekmat1\time_tot_Sp_hekmat1.mat');
t5=load('C:\Users\Hossein\Desktop\hekmat result\anjidani1\time_tot_Sp.mat');
t6=load('C:\Users\Hossein\Desktop\hekmat result\anjidani2\time_tot_Sp.mat');
t7=load('C:\Users\Hossein\Desktop\hekmat result\hadizadeh1\time_tot_Sp.mat');
t8=load('C:\Users\Hossein\Desktop\hekmat result\molavi1\time_tot_Sp.mat');
t9=load('C:\Users\Hossein\Desktop\hekmat result\molavi2\time_tot_Sp.mat');
t10=load('C:\Users\Hossein\Desktop\hekmat result\pajakh2\time_tot_Sp.mat');
% t11=load('C:\Users\Hossein\Desktop\hekmat result\sadeghi1\time_tot_Sp.mat');
t12=load('C:\Users\Hossein\Desktop\hekmat result\saidi1\time_tot_Sp.mat');



t1=t1.time_tot_Sp';
t2=t2.time_tot_Sp';
t3=t3.time_tot_Sp';
t4=t4.time_tot_Sp';  
t5=t5.time_tot_Sp'; % anjidani1
t6=t6.time_tot_Sp'; %anjidani2
t7=t7.time_tot_Sp'; % hadi zadeh
t8=t8.time_tot_Sp'; % molavi1
t9=t9.time_tot_Sp'; % molavi2
t10=t10.time_tot_Sp'; %pajakh
% t11=t11.time_tot_Sp'; % sadeghi
t12=t12.time_tot_Sp'; % saidi


load('C:\Users\Hossein\Desktop\hekmat result\filter coefficient\SOS.mat');
load('C:\Users\Hossein\Desktop\hekmat result\filter coefficient\G.mat');

channel = {Sig1_1, Sig1_2, Sig1_3, Sig2_1, Sig2_2, Sig3_1, Sig3_2, Sig_4_1, Sig_4_2, Sig_4_3, Sig_5_1 Sig_5_2, Sig_6_1, Sig_6_2, Sig_7_1...
    Sig_7_2, Sig_8_1, Sig_8_2, Sig_9_1, Sig_9_2, Sig_10_1, Sig_12_1, Sig_12_2};

t_total = {t1, t1, t1, t2, t2, t3, t3, t4, t4, t4, t5, t5, t6, t6, t7, t7, t8, t8, t9, t9, t10, t12, t12 };

data1=[];
data2=[];
for i=1:23
    signal  = channel{i}(1,:);%########
    timing1 = t_total{i}(1,:);%#########
    
    %  x=filtfilt(S2,num,den);
    fds1=[];fds2=[];fdk1=[];fdk2=[];fdh1=[];fdh2=[];en1=[];en2=[];power1=[];power2=[];
    entr1=[]; entr2=[];Lyapanuve1=[];Lyapanuve2=[];m1=[];m2=[];d1=[];d2=[];entr1=[];entr2=[];
    mean1=[];max1=[];min1=[];std1=[];prctile1=[];domain1=[];
    mean2=[];max2=[];min2=[];std2=[];prctile2=[];domain2=[];
    ZCR1=[];ZCRdensity1=[];ZCR2=[];ZCRdensity2=[];
    yy=detrend(signal);
    x=filtfilt(SOS,G,yy);
    
    for n=1:length(timing1)
        n
        s1=[];s2=[];
        timing2 =timing1+1;
        % s1=x(1,timing1(n)*fs+1:(timing1(n)+1)*fs);
        % s2=x(1,timing2(n)*fs+1:(timing2(n)+1)*fs);
        s1=x(1,timing1(n)*fs+1:timing1(n)*fs+1*fs); % mean spindle
        s2=x(1,timing2(n)*fs+1:timing2(n)*fs+1*fs);
        
        fds1(n)=fdsev(s1);
        fds2(n)=fdsev(s2);
        fdk1(n)=fdkatz(s1);
        fdk2(n)=fdkatz(s2);
        fdh1(n)=Higuchi(s1,8);  % not good
        fdh2(n)=Higuchi(s2,8);% 8= kmax
        en1(n)=energy(s1);
        en2(n)=energy(s2);
        power1(n)=f1(s1);
        power2(n)=f1(s2);
        fpeak1(n)=frequency_peak(s1);
        fpeak2(n)=frequency_peak(s2);
            [CI d1]=mutinf(s1',50);
            [CI d2]=mutinf(s2',50);
            m1=fnn(s1,8);
            m2=fnn(s2,8);
            Lyapanuve1(n)=lyap(s1,d1,m1);
            Lyapanuve2(n)=lyap(s2,d2,m2);
            %
            % %             AR1=lpc(s1,10);
            % %             AR2=lpc(s2,10);
            entr1(n)=entr(s1);
            entr2(n)=entr(s2);
% %             feature1(n)=AR1(2:end);
% %             feature2(n)=AR2(2:end);
%            domain1(n)=domain_theroshold(s1);
%            domain2(n)=domain_theroshold(s2);
            mean1(n)=mean(s1);
            mean2(n)=mean(s2);
            max1(n)=max(s1);
            max2(n)=max(s2);
            min1(n)=min(s1);
            min2(n)=min(s2);
            std1(n)=std(s1);
            std2(n)=std(s2);
%             prctile1(n)=prctile(x,75);
%             prctile2(n)=prctile(x,75);
            ZCR1(n)=ZCR(s1);
            ZCR2(n)=ZCR(s2);
            ZCRdensity1(n)=length(find(abs(diff(sign(diff(s1))))==2));
            ZCRdensity2(n)=length(find(abs(diff(sign(diff(s2))))==2));
        end
           data1=[data1 [fdh1;fds1;fdk1;Lyapanuve1;en1;power1;mean1;max1;min1;std1;prctile1;entr1;ZCR1;ZCRdensity1;domain1]]; 
           data2=[data2 [fdh2;fds2;fdk2;Lyapanuve2;en2;power2;mean2;max2;min2;std2;prctile2;entr2;ZCR2;ZCRdensity2;domain2]];
end

%% K-fold NN algorithm

Kfold=4; % K Fold
fold=280;% numbers in evry fold
zz=length(data1);
yy=0;
tp=zeros(1,Kfold);
tn=zeros(1,Kfold);

for ss=1:Kfold
    ss
    test1=data1(:,fold*(ss-1)+1:ss*fold); %spindle% jaye ss adad beazi malome % testi hast ke 10 fold azash bardashtim
    test2=data2(:,fold*(ss-1)+1:ss*fold);% non spindle
    p1=[data1(:,1:fold*(ss-1)) data1(:,fold*ss+1:zz)]; %10 ta ro az kolesh hazf mikone
    p2=[data2(:,1:fold*(ss-1)) data2(:,fold*ss+1:zz)];
    p=[p1 p2];
    t=[ones(1,zz-fold) zeros(1,zz-fold)]; % 10 fold hazf kardam az lable ha.
    nu=randperm((zz-fold)*2);
    
    for fn=(1:zz-fold)*2
        pp(fn,:)=p(:,nu(fn));% dataye rand per khorde hast.
        tt(fn)=t(nu(fn));% t lable haye randperm khorde hast
    end
    
          net=newff(pp',tt,7);
%     net = feedforwardnet(7);
    net.layers{1}.transferFcn = 'logsig';
%          view(net);
    [net tr]=train(net,pp',tt);
    while tr.perf>0.075
        [net tr]=train(net,pp',tt);
    end
    sa1(ss,:)=sim(net,test1);
    sa2(ss,:)=sim(net,test2);
    yy(ss)=0;
    for w=1:fold % 10 fold hast
        if sa1(w)>=0.5
            tp(ss)=tp(ss)+1;
            yy(ss)=yy(ss)+1;
        end
        if sa2(w)<=0.5
            tn(ss)=tn(ss)+1;
            yy(ss)=yy(ss)+1;
        end
    end
%         uu=sprintf('f%g',ss);
%         save(uu,'net')
    per(ss)=yy(ss)/(2*fold)
end
accuracy=mean(per)
sensi=mean(tp)/fold
spe=mean(tn)/fold
%%%%%%%%%%%%%%%%% KNN Algorithm

for ss=1:Kfold
    ss
    test1=data1(:,fold*(ss-1)+1:ss*fold); %spindle% jaye ss adad beazi malome % testi hast ke 10 fold azash bardashtim
    test2=data2(:,fold*(ss-1)+1:ss*fold);% non spindle
    p1=[data1(:,1:fold*(ss-1)) data1(:,fold*ss+1:zz)]; %10 ta ro az kolesh hazf mikone
    p2=[data2(:,1:fold*(ss-1)) data2(:,fold*ss+1:zz)];
    p=[p1 p2];
    t=[ones(1,zz-fold) zeros(1,zz-fold)]; % 10 fold hazf kardam az lable ha.
   
                
label=genlab([339 339],[1 0]');
data=[data1 data2]';

label1=genlab([297],[1]');
A1= dataset(p1',label1);
% [B1,C1] = gendat(A1,0.99);
label1t=genlab([42],[1]');
B1= dataset(p1',label1);
C1= dataset(test1',label1t);

label2=genlab([297],[0]');
A2= dataset(p2',label2);
% [B2,C2] = gendat(A2,0.99);
label2t=genlab([42],[0]');
B2= dataset(p2',label2);
C2= dataset(test2',label2t);


A=[A1;A2];
B=[B1;B2];
C=[C1;C2];
w1 = svc([],'p',1);
w1 = setname(w1,'svc poly');

w2 = svc([],'r',1);
w2 = setname(w2,'svc RBF');

w3 = svc([],'e',1);
w3 = setname(w3,'svc exponential');

w4 = knnc([],1);
w4 = setname(w4,'1-NN');

w5 = knnc([],2);
w5 = setname(w5,'2-NN ');

w6 = klm([],0.95)*ldc;
w6 = setname(w6,'klm - ldc');

w7 = ldc;
w7 = setname(w7,'ldc');

w8 = knnc([],1);
w8 = setname(w8,'1-NN');
% Store classifiers in a cell
W = {w4};
% Train them all
V = B*W;
% Test them all
disp([newline 'Errors for individual classifiers'])
% testc(B,V)
testc(C,V)
end

%%%%%%%%%%%%%%%% check for spindle detection in signal

