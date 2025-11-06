clc,close all
res=imread('1_2 (379).jpg');
res=imresize(res,[224,224]);
sayac=1;
for i=1:28:224-28
    for j=1:28:224-28
        exm=res(i:i+55,j:j+55,:);
        subplot(7,7,sayac),imshow(exm)
        title(dr(sayac),'FontSize',24)
        sayac=sayac+1;
    end
end
figure,imshow(res)
