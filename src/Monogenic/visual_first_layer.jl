# output: the output of the first Layer
# index: the index of the dataset

function visual_first_layer(output, index = 1)

	if length(size(output)) == 4

		nchannel = 3;
		scale = Int(size(output, 3) / nchannel);

		fig = PyPlot.figure(figsize=(5, 10))

		for scal = scale:-1:1

		    o1 = output[:,:,1 + 3*(scal - 1),index];
		    o2 = output[:,:,2 + 3*(scal - 1),index];
		    o3 = output[:,:,3 + 3*(scal - 1),index];
		    
		    if scal == 1
		    	subplot(scale, nchannel, 1 + 3*(scal - 1), title = "1", 
		    	ylabel = string(scal), xticks = [], yticks = []); global p = imshow(o1, cmap="gray"); 
		    	subplot(scale, nchannel, 2 + 3*(scal - 1), title = "i"); imshow(o2, cmap="gray"); axis("off"); 
		    	subplot(scale, nchannel, 3 + 3*(scal - 1), title = "j"); imshow(o3, cmap="gray"); axis("off"); 
		    else
		    	subplot(scale, nchannel, 1 + 3*(scal - 1), 
		    	ylabel = string(scal), xticks = [], yticks = []); imshow(o1, cmap="gray"); 
		    	subplot(scale, nchannel, 2 + 3*(scal - 1)); imshow(o2, cmap="gray"); axis("off"); 
		    	subplot(scale, nchannel, 3 + 3*(scal - 1)); imshow(o3, cmap="gray"); axis("off"); 
			end

		end

		fig.text(0.05,0.5, "scale", ha="center", va="center", rotation=90)
		suptitle("First Layer Output");
		fig.subplots_adjust(right=0.8)
		cbar = fig.add_axes([0.85, 0.15, 0.01, 0.7])
		fig.colorbar(p, cax = cbar);



	else
		print("Wrong Dimension. The input is 4 dimensional.\n")
	end

end