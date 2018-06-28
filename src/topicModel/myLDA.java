package topicModel;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.ArrayList;
import javafx.util.Pair;
import cc.mallet.util.Randoms;

import java.util.stream.*;

public class myLDA {
	
	// Helper classes
	private class Word {
		int word_idx;
		int topic_assignment;
		Word(int word_idx, int topic_assignment) {
			this.word_idx = word_idx;
			this.topic_assignment = topic_assignment;
		}
	}
	
	private class Document {
		// nm, store number of words assigned to each topic per document
		int[] topicCounts;
		int wordCount;  // i.e. sum of topicCounts
		
		// latent variable -- topic proportions for this doc
		double theta[];
		
		// actual words in doc
		// storing them this way is less space efficient (we'd prefer to keep sparse representation)
		// but it's much easier to iterate in parallel with current assignments if they are in sync
		List<Word> words;

		Randoms rand = new Randoms();
		
		// This is a more space efficient way
//		// current assignments for each word in this doc
//		List<Integer> currentAssignments;

//		// Actual words (Sparse BOW format)
//		// dim 1 = word
//		// dim 2 = 2D array, 1st element = word idx, 2nd element = word count
//		int[][] words;
		
		String docID;
		
		Document(String[] input_words) {
			topicCounts = new int[num_topics];
			
			this.docID = input_words[0];
			words = new ArrayList<Word>();
			
			// start at 1 b/c first word is docID
			for (int i = 1; i < input_words.length; i++) {
				String[] parts = input_words[i].split(":");
				assert(parts.length == 2);
				int word_idx = Integer.parseInt(parts[0]);
				int num_of_this_word = Integer.parseInt(parts[1]);
				for (int j = 0; j < num_of_this_word; j++) {
					// do initialization, randomly sample a topic
					int topic_assignment = rand.nextInt(num_topics);
					words.add(new Word(word_idx, topic_assignment));
					
					// increment doc counts
					wordCount++;
					topicCounts[topic_assignment]++;
					
					// increment global counts (WARNING not great programming practice)
					wordsPerTopic[topic_assignment][word_idx]++;
					totalTopicCount[topic_assignment]++;
				}
			}
			// during initialization, also do global sanity checks
			this.sanity_check();
			for (int i = 0; i < num_topics; i++) {
				assert(IntStream.of(wordsPerTopic[i]).sum() == totalTopicCount[i]);
			}
		}
		
		void resample() {
			// for each word, undo it's current assignment
			for (Word word : words) {
				topicCounts[word.topic_assignment]--;
				wordCount--;
				wordsPerTopic[word.topic_assignment][word.word_idx]--;
				totalTopicCount[word.topic_assignment]--;
				
				// Calculate probability that word belongs with each topic
				double[] topic_probs = new double[num_topics];
				for (int i = 0; i < num_topics; i++) {
					// number of times words in this doc are assign to topic i
					// In Darling notes, we normalize this, but normalization term
					// is same for all docs so I think we're fine
					double right = topicCounts[i] + alpha;
					
					// number of times this topic occurred with this word (type, ignore current token)
					// normalize by number of times topic occurred with any word
					double left = (wordsPerTopic[i][word.word_idx] + beta) / (totalTopicCount[i] + (vocab_size * beta));
					topic_probs[i] = left * right;
				}
				// assign to a new topic
				double sum = DoubleStream.of(topic_probs).sum();
				word.topic_assignment = rand.nextDiscrete(topic_probs, sum);
				
				// update counts with new topic
				topicCounts[word.topic_assignment]++;
				wordCount++; //[this is a little redundant but whatever]
				wordsPerTopic[word.topic_assignment][word.word_idx]++;
				totalTopicCount[word.topic_assignment]++;
			}
		}
		
		// check invariants
		void sanity_check() {
			assert(IntStream.of(topicCounts).sum() == wordCount);
			assert(words.size() == wordCount);			
		}
		
		void print_doc() {
			System.out.println(docID);
			for (Word word : words) {
				System.out.println(idx_to_vocab[word.word_idx] + " " + word.topic_assignment);
			}
			System.out.println();
		}
		
		void set_theta() {
			theta = new double[num_topics];
			int num_words = words.size();
			for (int i = 0; i < num_topics; i++)
				theta[i] = (topicCounts[i] + alpha) / (num_words + (alpha * vocab_size)); 
		}

	}
	// *******************************************************************************************************************************
	// END HELPER CLASSES
	// *******************************************************************************************************************************
	int num_topics;
	double alpha;
	double beta;
	int max_iter;
	String[] idx_to_vocab;
	int num_docs;
	int vocab_size;
	

	// nk, store number of times each word appears with each topic
	int[][] wordsPerTopic;
	int[] totalTopicCount;
	List<Document> docs = new ArrayList<Document>();
	
	// latent variables
	double phi[][];

	
	public myLDA(int num_topics, double alpha, double beta, int max_iter) {
		this.alpha = alpha;
		this.beta = beta;
		this.max_iter = max_iter;
		this.num_topics = num_topics;
		
	}
	
	// debugging
	void sanity_check() {
		assert(num_docs == docs.size());
		for (int i = 0; i < num_topics; i++) {
			assert(IntStream.of(wordsPerTopic[i]).sum() == totalTopicCount[i]);
		}
		for (Document d: docs) {
			d.sanity_check();
		}
	}
	
	void print_model() {
		System.out.println(vocab_size);
		System.out.println(num_topics);
		
		for (int i = 0; i < 10; i ++)
			System.out.println(idx_to_vocab[i]);
		
		for (int i = 0; i < 10; i ++)
			docs.get(i).print_doc();
	}
	
	// Read in data
	void readVocab(String vocabFilename) {
		vocab_size = 0;
		try {
			FileInputStream inStream = new FileInputStream(vocabFilename);

			// silly but we're just going to iterate twice b/c we need vocab size
			// to initialize array
			BufferedReader in1 = new BufferedReader(new InputStreamReader(inStream));
			while (in1.readLine() != null)
				vocab_size += 1;
			
			// Restart for real this time
			inStream.getChannel().position(0);
			in1 = new BufferedReader(new InputStreamReader(inStream));
			
			idx_to_vocab = new String[vocab_size];
			String str1;
			int idx = 0;
			while ((str1 = in1.readLine()) != null) {
				idx_to_vocab[idx] = str1.trim();
				idx++;
			}
			in1.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	void readData(String docFilename, String vocabFilename) {
		// we need to know vocab size, so read in vocabFile first
		readVocab(vocabFilename);
		
		assert (vocab_size != 0);
		wordsPerTopic = new int[num_topics][vocab_size];
		totalTopicCount = new int[num_topics]; // total number of words assigned to each topic
		
		// now read documents
		try {
			BufferedReader in1 = new BufferedReader(new InputStreamReader(
					new FileInputStream(docFilename)));
			String str1;
			while ((str1 = in1.readLine()) != null) {
				String[] mainparts = str1.split("\t");
				docs.add(new Document(mainparts));
				num_docs++;
			}
			in1.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		sanity_check();
	}
	
	// Compute likelihood
	
	
	// Do Gibbs sampling
	void do_sampling() {
		for (int i = 0; i < max_iter; i++) {
			for (Document doc : docs) {
				doc.resample();
			}
			sanity_check();
		}
		calculate_parameters();
	}
	
	void calculate_parameters() {
		// latent variables
		phi = new double[num_topics][vocab_size];
		for (int i = 0; i < num_topics; i++) {
			for (int j = 0; j < vocab_size; j++)
				phi[i][j] = (wordsPerTopic[i][j] + beta) / (totalTopicCount[i] + (beta * vocab_size));
		}
		
		for (Document doc : docs) {
			doc.set_theta();
		}
		
	}
	
	void print_topics(int num_to_print) {
		for (int i = 0; i < num_topics; i++) {
			System.out.println("Topic " + i);
			double[] current_phi = phi[i];
			Stream<Integer> sorted_idx =  IntStream.range(0, phi[i].length)
					.boxed().sorted((j, k) -> Double.compare(current_phi[k], current_phi[j]));
			sorted_idx.limit(num_to_print).forEach(idx -> System.out.println(idx_to_vocab[idx]));
			System.out.println();			
		}
	}
	
	// Inputs: 
	//    vocab file, where line 0 has word 0
	//    Document/term matrix file where each line represents a document and contains [word_idx]:[word_count] (first token in line is document id)
	//    alpha value
	//    beta value
	//    max number of iterations
	//    number of topics
	
	public static void main(String args[]) {
		System.out.println("Initialize Model");
		
		// TODO: make these cmd line parameters
		double alpha = 0.01;
		double beta = 0.01;
		int num_topics = 50;
		int max_iter = 10000;
		
		String docfileName = "/Users/anjaliefield/plot_summaries_test.tsv";
		String vocabfileName = "/Users/anjaliefield/vocab_test.tsv";
		
		myLDA tester = new myLDA(num_topics, alpha, beta, max_iter);
		tester.readData(docfileName, vocabfileName);
		
		//tester.print_model();
		System.out.println("Starting LDA");
		tester.do_sampling();
		tester.print_topics(15);
	}
}