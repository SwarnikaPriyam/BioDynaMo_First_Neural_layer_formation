// -----------------------------------------------------------------------------
//
// Copyright (C) 2021 CERN & Newcastle University for the benefit of the
// BioDynaMo collaboration. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// See the LICENSE file distributed with this work for details.
// See the NOTICE file distributed with this work for additional information
// regarding copyright ownership.
//
// -----------------------------------------------------------------------------
#ifndef NEURAL_LAYER_DEVELOP_H_
#define NEURAL_LAYER_DEVELOP_H_

#include "biodynamo.h"

#include <iostream>
#include <ctime>
#include <string>
#include <vector>

#include "core/container/math_array.h"
#include "core/diffusion/diffusion_grid.h"
#include "core/resource_manager.h"
#include "core/simulation.h"
#include "core/util/random.h"

#include "core/substance_initializers.h"
#include "Math/DistFunc.h"
#include "core/diffusion/diffusion_grid.h"

namespace bdm {

  // Define my custom cell class MyCell, which extends Cell by adding extra data
  // members: cell_color, nr_div, cell_type, grn_state
  class MyCell : public Cell {  // our object extends the Cell object
    // create the header with our new data member
    BDM_AGENT_HEADER(MyCell, Cell, 1);
 
  public:
    MyCell() {}
    explicit MyCell(const Double3& position, int cell_type, int grn_state, int nr_divisions, std::string guidance_cue )
    : Base(position), cell_type_(cell_type), grn_state_(grn_state), nr_divisions_(nr_divisions), guidance_cue_(guidance_cue) {}
    virtual ~MyCell() {}

    // If MyCell divides, the daughter has to initialize its attributes
    void Initialize(const NewAgentEvent& event) override {
      Base::Initialize(event);
      if (auto* mother = dynamic_cast<MyCell*>(event.existing_agent)) {
        if (event.GetUid() == CellDivisionEvent::kUid) {                                  // If the event that this object has undergone is a cell division event,
                                                                                          // then set no. of divisions for mother cell as initial number of divisions incremented by 1. 
        
          std::cout << "init: mother ID / grn_state_ / cell_type_ : " << mother->GetUid() << " / " << mother->grn_state_ << " / " << mother->cell_type_ << std::endl;
          std::cout << "init: daughter ID / grn_state_ / cell_type_: " << this->GetUid() << " / " << this->grn_state_ << " / " << this->cell_type_ << std::endl;
          
          n_ = mother->GetNrDivisions();
          std::cout << "Value of n: " << n_ << std ::endl;
          mother->SetNrDivisions(mother->GetNrDivisions()+1);
          nr_divisions_=mother->nr_divisions_;
          std::cout << "Number of divisions: " << nr_divisions_ << std ::endl;
          
        }
      }
    }

  // getter and setter for our new data member
  void SetCellColor(int cell_color) { cell_color_ = cell_color; }
  int GetCellColor() const { return cell_color_; }

  int GetNrDivisions() { return nr_divisions_; }
  void SetNrDivisions(int nr_div) { nr_divisions_ = nr_div;}
  
  int GetCellType() const { return cell_type_; }
  void SetCellType(int cell_type) { cell_type_ = cell_type; }
  
  int GetGRNState() const { return grn_state_; }
  void SetGRNState(int grn_state) {grn_state_ = grn_state; }
  
  std::string GetGuidanceCue() const { return guidance_cue_; }
  int cell_type_;
  int grn_state_;


private:

  // declare new data member and define their type
  // private data can only be accessed by public function and not directly
  bool can_divide_;
  int nr_divisions_;
  int cell_color_;
  int n_;
  
  std::string guidance_cue_;

};

// Define growth behaviour
struct Growth : public Behavior {                                 // Behavior : The cell grows and divides here
  BDM_BEHAVIOR_HEADER(Growth, Behavior, 1);                       // A new behavior structure Growth is defined that is applied to cell elements

  Growth() { AlwaysCopyToNew(); }                                 // This Growth is copied into the cell daughter (so the daughter will also contain an instance of the behavior Growth)
  virtual ~Growth() {}
  
  double speed_=1.0;
  DiffusionGrid* dgrid_ = Simulation::GetActive()->GetResourceManager()->GetDiffusionGrid("GuidanceCue");
  
  void Run(Agent* agent) override {                               // Run Method executed every time step

    if (auto* cell = dynamic_cast<MyCell*>(agent)) {
    
    auto& position = cell->GetPosition();
    double currConc = dgrid_->GetConcentration(position);
    

      if (cell->GetDiameter() < 15) {                             // If diameter is less than 15, the cell will grow
        //auto* random = Simulation::GetActive()->GetRandom();
        // Here 700 is the speed and the change to the volume is based on the
        // simulation time step.
        // The default here is 0.01 for timestep, not 1.
        cell->ChangeVolume(700);

      } else {  

        if (cell->GetNrDivisions()<8) {                          // If number of divisions is less than 8, then cell will divide
          cell->Divide();
          std::cout << "Division happens" << std ::endl;
          std::cout << "conc: " << currConc << " " << std::endl;
          cell->SetCellColor(9);                                  // Change red color
          cell->SetCellType(1);
          cell->SetGRNState(1);

        } else if (cell->GetGRNState()==0 && cell->GetCellType()==0){
               //std::cout << "conc: " << currConc << " " << std::endl;
               if (currConc>0.0132053) {
                  std::cout << "stopped a cell of type " << 0 << " at conc: "<<  currConc << std::endl;
                  cell->SetCellColor(5);                           // Change green color
                  cell->RemoveBehavior(this);
                } else {
                 auto& position = cell->GetPosition();
                 Double3 gradient;
                  dgrid_->GetGradient(position, &gradient);
                  gradient[0]=0.0;
                  gradient[1]=0.0;
                  if (abs(gradient[2])>0) {
                    gradient[2]=gradient[2]/gradient[2];
                    }
                  cell->UpdatePosition(gradient * speed_);
                  }
         }
      }
    }
 }

};

  template <typename Function>
  static void Grid2D(size_t agents_per_dim, double space,
    Function agent_builder) {
      #pragma omp parallel
      {
        auto* sim = Simulation::GetActive();
        auto* ctxt = sim->GetExecutionContext();

        #pragma omp for
        for (size_t x = 0; x < agents_per_dim; x++) {
          auto x_pos = x * space;
          for (size_t y = 0; y < agents_per_dim; y++) {
            auto y_pos = y * space;
            //for (size_t z = 0; z < agents_per_dim; z++) {
            auto* new_agent = agent_builder({x_pos, y_pos, 0.0});
            new_agent->SetMass(0.0001);
            ctxt->AddAgent(new_agent);
            //}
          }
        }
      }
    } 

// Define the function : PoissonBandAtPos()

  /// An initializer that follows a Poisson (normal) distribution along one axis
  /// In contrast to the previous function, the location of the peak can be set
  /// The function ROOT::Math::poisson_pdfd(X, lambda) follows the normal
  /// probability density function:
  /// {e^( - lambda ) * lambda ^x )} / x!
  class PoissonBandAtPos {
  public:
    /// @brief      The constructor
    ///
    /// @param[in]  lambda The lambda of the Poisson distribution
    /// @param[in]  axis   The axis along which you want the Poisson distribution
    ///                    to be oriented to
    ///
    //PoissonBandAtPos(double lambda, uint8_t axis, Double3 position) {
    PoissonBandAtPos(double lambda, uint8_t axis, std::array<double, 3> position) {
      lambda_ = lambda;
      axis_ = axis;
      pos_x_ = position[0];
      pos_y_ = position[1];
      pos_z_ = position[2];
    }

    /// @brief      The model that we want to apply for substance initialization.
    ///             The operator is called for the entire space
    ///
    /// @param[in]  x     The x coordinate
    /// @param[in]  y     The y coordinate
    /// @param[in]  z     The z coordinate
    ///
    double operator()(double x, double y, double z) {
      switch (axis_) {
        case Axis::kXAxis:
        return ROOT::Math::poisson_pdf(x-pos_x_, lambda_);
        case Axis::kYAxis:
        return ROOT::Math::poisson_pdf(y-pos_y_, lambda_);
        case Axis::kZAxis:
        return ROOT::Math::poisson_pdf(z-pos_z_, lambda_);
        default:
        throw std::logic_error("You have chosen a non-existing axis!");
      }
    }

  private:
    double lambda_;
    uint8_t axis_;
    double pos_x_;
    double pos_y_;
    double pos_z_;
  };
  
// List the extracellular substances
enum Substances { GuidanceCue };
                        
inline int Simulate(int argc, const char** argv) {                // Simulation function defined  
  auto set_param = [](Param* param) {                             // Create lambda function to set the simulation parameters programmatically
    param->bound_space = true;
    param->min_bound = 0;
    param->max_bound = 200;  // cube of 200*200*200
    param->unschedule_default_operations = {"mechanical forces"};
  };

  Simulation simulation(argc, argv, set_param);                   // Create BioDynaMo Simulation - simulation object created
  auto* rm = simulation.GetResourceManager();                     // Initiate resource manager which stores the agents; obtain reference to this object afterwards
  
  size_t nb_of_cells = 1;                                         // number of cells in the simulation to start with, in this case only 1
  size_t diameter = 10;                                           // diameter of the cell
  size_t spacing = 25;                                            // spacing required for Grid2D function
  
  // Define the substances that cells may secrete
  ModelInitializer::DefineSubstance(GuidanceCue, "GuidanceCue", 0.0, 0, 10);
  //ModelInitializer::InitializeSubstance(GuidanceCue, PoissonBand(1, Axis::kZAxis));
  //Double3 gradPos = {0.,0.,3.};
  std::array<double, 3> subPos = {0., 0., 3.};
  ModelInitializer::InitializeSubstance(GuidanceCue, PoissonBandAtPos(100, Axis::kZAxis, subPos));       // Guidance cue is a gradient along the vertical axis.
                                                                                                         // Here, such an extracellular gradient is created first.
  
  // To define how cells will look like we will create a construct in the
  // form of a C++ lambda as follows.
  auto construct = [&](const Double3& position) {                          // Construct lambda defines the properties (physical properties like diameter, color, mass  
                                                                           // or biological properties and behaviors like guidance cue, chemotaxis, substance secretion) of each cell that we create
  
    // creating the cell at position x, y, z                               // Create a cell object; cell class is the most basic cell class present in the standard library of BioDynaMo
    MyCell* cell = new MyCell(position,0,0,0,"GuidanceCue");               // passing values 0, 0, 0 to cell_type, grn_state, nr_divisions
    
    // set cell parameters
    cell->SetDiameter(diameter);
    cell->SetCellColor(0);                                                 // Set blue color
    cell->SetMass(0.00001);
    cell->AddBehavior(new Growth());                                       // Create a new cell that contains Growth behavior in the Simulate method
    
    return cell;
  };
  
  Grid2D(nb_of_cells, spacing, construct);

  int n;
  for (int i=0; i<2000; i++){

    // Run simulation for one timestep
    simulation.GetScheduler()->Simulate(1);
    n = rm->GetNumAgents();
    std::cout << "Number of cells: " << n << std ::endl;
  }
  
  // The number of Type 1 and Type 2 cells
  int num_cells_Type1 = 0;
  int num_cells_Type2 = 0;
    
  rm->ForEachAgent([&](Agent* agent) {                                     // Loop through all the cells to count the number of Type 1 and Type 2 cells
    if (auto* cell = dynamic_cast<MyCell*>(agent)) {
      auto type = cell->GetCellType(); 
      //std::cout << "Cell Type: " << type << std::endl;  
      if (type == 1) {
          num_cells_Type2++;        
      } else if (type == 0){
            num_cells_Type1++;
      }
    }
  });

  std::cout << "Number of Type1 cells: " << num_cells_Type1 << std::endl;
  std::cout << "Number of Type2 cells: " << num_cells_Type2 << std ::endl; 
  std::cout << "Simulation completed successfully!" << std::endl;
  return 0;
}

}  // namespace bdm

#endif  // NEURAL_LAYER_DEVELOP_H_
