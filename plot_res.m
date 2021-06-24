y = A(1,1);
x=1:500;
semilogy(x,y*ones(size(x)))
hold on

fileID = fopen('CGresiduals.txt','r');
formatSpec = '%f %d';
sizeA = [2 Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
xCG = A(2,:);
yCG = A(1,:);
semilogy(xCG,yCG,"r*");
hold on

fileID = fopen('GMRESresiduals.txt','r');
formatSpec = '%f %d';
sizeA = [2 Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
xGMRES = A(2,:);
yGMRES = A(1,:);
semilogy(xGMRES,yGMRES,"k.");
hold on

fileID = fopen('GMRESrestart.txt','r');
formatSpec = '%f %d';
sizeA = [2 Inf];
A = fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
xrestart = A(2,:);
yrestart = A(1,:);
semilogy(xrestart,yrestart,"x");

curtick = get(gca, 'xTick');
xticks(unique(round(curtick)));
xlabel('Iteration')
ylabel('Norm of Residual')
legend('Gaussian elimination','CG','GMRES');
saveas(gcf,'struc500','epsc')


