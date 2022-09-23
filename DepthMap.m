close all;
%%%Read Image stack
ImgStack=load("stack.mat");
fNames = fieldnames(ImgStack);
NumberOfFrames=length(fNames)-2;
%Taking each frames
for n = 1:NumberOfFrames
    frames(:,:,n)=ImgStack.(fNames{n+1});
end
%Kernal for finding X gradiant
Xgradiant=[0,0,0;1,-2,1;0,0,0];
%Kernal for finding Y gradiant
Ygradiant=[0,1,0;0,-2,0;0,1,0];
%Taking each frame and convolving to find gradiants
for n = 1:NumberOfFrames
    XgradiantFrames(:,:,n)=conv2(frames(:,:,n),Xgradiant,'same');
    YgradiantFrames(:,:,n)=conv2(frames(:,:,n),Ygradiant,'same');
    %taking modulus and adding to form modified laplacian
    ML(:,:,n)=abs(XgradiantFrames(:,:,n))+abs(YgradiantFrames(:,:,n));
end
%Finding the depth of image for 1*1 neighbourhood
SizeOfNeighbourhood=0;
[dbar1,MaximumsFrame]=FindDepth(ML,SizeOfNeighbourhood,NumberOfFrames);
%%%Display Depth map
figure('name',"depth map for 1*1");
surf(dbar1)
%Finding the depth of image for 3*3 neighbourhood
SizeOfNeighbourhood=1;
dbar2=FindDepth(ML,SizeOfNeighbourhood,NumberOfFrames);
%%%Display Depth map
figure('name',"depth map for 3*3");
surf(dbar2)
%Finding the depth of image for 5*5 neighbourhood
SizeOfNeighbourhood=2;
dbar3=FindDepth(ML,SizeOfNeighbourhood,NumberOfFrames);
%%%Display Depth map
figure('name',"depth map for 5*5");
surf(dbar3)

AllFocusedImage=ones(115,115);
for x=1:115
        for y=1:115
            AllFocusedImage(x,y)=frames(x,y,MaximumsFrame(x,y));
        end
end
figure('name',"focused");
imshow(uint8(AllFocusedImage))

%%Function to find the SML and estimate the depth.
function [dbar,MaximumsFrame] =FindDepth(ML,SizeOfNeighbourhood,NumberOfFrames)
%Initializing the SML stack
SML=zeros(115,115,NumberOfFrames);
for n = 1:NumberOfFrames                  %Traversing through each frames
    for x=1:115                           %Traversing through each pixels in the frame
        for y=1:115
            %%%Summing the modified laplacian in the neighbourhood to get
            %%%SML
            for i=x-SizeOfNeighbourhood:x+SizeOfNeighbourhood
                for j=y-SizeOfNeighbourhood:y+SizeOfNeighbourhood
                    if i>0 && j>0 && j<116 && i<116
                    SML(x,y,n)=SML(x,y,n)+ML(i,j,n);
                    end
                end
            end
        end
    end
end
Maximums=zeros(115,115);            %Initializing matrix to store the corresponding maximums of each pixel stack
AfterMaximums=zeros(115,115);       %Initializing matrix to store the SML value after the maximum of each pixel stack
BeforeMaximums=zeros(115,115);      %Initializing matrix to store the SML value before the maximum of each pixel stack
MaximumsFrame=zeros(115,115);       %Initializing matrix to store the frame number of maximums of each pixel stack
for x=1:115
        for y=1:115
           % Finding the maximum value of SML
            Maximums(x,y)=max(SML(x,y,:),[],'all');
            %Finding the frame number of maximum
            for n = 1:NumberOfFrames
                if SML(x,y,n)==Maximums(x,y)
                    MaximumsFrame(x,y)=n;
                    break;
                    
                end
            end
            %Finding values after and before the maximum
            if MaximumsFrame(x,y)==100 
                AfterMaximums(x,y)=SML(x,y,MaximumsFrame(x,y));
                BeforeMaximums(x,y)=SML(x,y,MaximumsFrame(x,y)-1);
            elseif MaximumsFrame(x,y)==1 
                AfterMaximums(x,y)=SML(x,y,MaximumsFrame(x,y)+1);
                BeforeMaximums(x,y)=SML(x,y,MaximumsFrame(x,y));
            else
                AfterMaximums(x,y)=SML(x,y,MaximumsFrame(x,y)+1);
                BeforeMaximums(x,y)=SML(x,y,MaximumsFrame(x,y)-1);
            end
        end
end
deltaD=50.50;
dbar=zeros(115,115);%Initializing the matrix to store depth value.
%Gaussian fitting and finding the depth value.
for x=1:115
        for y=1:115
            dbar(x,y)=(   log(Maximums(x,y))-log(BeforeMaximums(x,y)))*(( ( ( MaximumsFrame(x,y)+1)*deltaD)^2)-(( MaximumsFrame(x,y)*deltaD)^2));
            dbar(x,y)=dbar(x,y)-( log(Maximums(x,y))-log(AfterMaximums(x,y)))*(( ((MaximumsFrame(x,y)-1)*deltaD)^2 ) -((MaximumsFrame(x,y)*deltaD)^2));
            dbar(x,y)=dbar(x,y)/(2*deltaD*(2*log(Maximums(x,y))-log(AfterMaximums(x,y))-log(BeforeMaximums(x,y))));
        end
end
end