# output: the output of the first Layer
# j: the index of the dataset

function visual_second_layer(output, index = 1)

	if length(size(output)) == 5

		nchannel = 3;

		n_layer1 = size(output, 4);
		n_layer2 = size(output, 3);


		scale_layer1 = Int(n_layer1 / nchannel);
		scale_layer2 = Int(n_layer2 / nchannel);


		fig = PyPlot.figure(figsize=(10, 10))

		for scal2 = scale_layer2:-1:1
			for scal1 = scale_layer1:-1:1

				# index within the block
				for k = nchannel:-1:1 # row in block
					for j = nchannel:-1:1 # column in block
					    o1 = output[:,:, k + nchannel*(scal2 - 1),j + nchannel*(scal1 - 1), index];

					    # first row
					    bool_1st_row = false;
					    for jj = 1:nchannel
					    	if (scal2 == 1) & (k == 1) & (j == jj)

					    		if jj == 1
					    			title_name = "(1,:)";
					    		elseif jj == 2
					    			title_name = "(i,:)";
					    		elseif jj == 3
					    			title_name = "(j,:)";
					    		end

					    		subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], title = title_name); imshow(o1, cmap="gray"); 
					    		bool_1st_row = true;
					    	end
					    end

					    # last column
					    bool_last_column = false;
					    for jj = 1:nchannel
					    	if (scal1 == scale_layer1) & (j == nchannel) & (k == jj)
					    
					    		if jj == 1
					    			title_name = "(:,1)";
					    		elseif jj == 2
					    			title_name = "(:,i)";
					    		elseif jj == 3
					    			title_name = "(:,j)";
					    		end
					    		pp = subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], ylabel = title_name); 
					    		pp.yaxis.set_label_position("right")
					    		imshow(o1, cmap="gray"); 
					    		bool_last_column = true;
					    	end
					    end

					    if (bool_last_column == false) & (bool_1st_row == false)
					    #if bool_1st_row == false
					    	subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = []); imshow(o1, cmap="gray"); 
					    end					    

					    # first column
					    if (scal1 == 1) & (j == 1)
					    	subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], ylabel = Int(scal2)); imshow(o1, cmap="gray"); 
					    end

					    # last row
					    if (scal2 == scale_layer2) & (k == nchannel)
					    	subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], xlabel = Int(scal1)); imshow(o1, cmap="gray"); 
					    end

					    # (1,1)
					    if (scal1 == 1) & (scal2 == 1) & (j == 1) & (k == 1)
					    	subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], title = "(1,:)", ylabel = 1); global p = imshow(o1, cmap="gray"); 
					    end

					    # (1,1ast)
					    if (scal1 == scale_layer1) & (scal2 == 1) & (j == nchannel) & (k == 1)
					    	pp = subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], title = "(j,:)", ylabel = "(:,1)"); 
					    	pp.yaxis.set_label_position("right");
					    	imshow(o1, cmap="gray"); 
					    	bool_last_column = true; 
					    end

					    # (last,1)
					    if (scal1 == 1) & (scal2 == scale_layer2) & (j == 1) & (k == nchannel)
					    	subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], xlabel = 1, ylabel = scal2); imshow(o1, cmap="gray"); 
					    end


					    # (last,1ast)
					    if (scal1 == scale_layer1) & (scal2 == scale_layer2) & (j == nchannel) & (k == nchannel)
					    	pp = subplot(n_layer2, n_layer1,  n_layer2 * ((k - 1) + nchannel*(scal2 - 1)) +  j + nchannel*(scal1 - 1) , xticks = [], yticks = [], xlabel = scal1, ylabel = "(:,j)"); 
					    	pp.yaxis.set_label_position("right");
					    	imshow(o1, cmap="gray"); 
					    	bool_last_column = true; 
					    end					    

					end
				end

	 


			end
		end

		fig.text(0.5,0.05, "first layer index", ha="center", va="center")
		fig.text(0.08,0.5, "second layer index", ha="center", va="center", rotation=90)
		suptitle("Second Layer Output");
		fig.subplots_adjust(right=0.8)
		cbar = fig.add_axes([0.85, 0.15, 0.01, 0.7])
		fig.colorbar(p, cax = cbar);




	else
		print("Wrong Dimension. The input is 5 dimensional.\n")
	end

end
