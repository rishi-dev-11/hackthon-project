import React, { createContext, useContext, useState, useEffect } from 'react';
import { getCurrentUser } from '../services/api';

// Create context
export const SubscriptionContext = createContext({
  userTier: 'free',
  canUseFeature: () => false,
  getFeatureLimit: () => 0,
  getFeatureDetails: () => ({}),
  canUseTemplate: () => false,
  switchPlan: () => {},
  isDevMode: false,
});

// Features and limits for different user tiers
const tierFeatures = {
  free: {
    maxDocuments: 5,
    maxTemplates: 3,
    llmEnabled: false,
    multiLanguage: false,
    ocrLanguages: ['eng'],
    styleGuideCompliance: false,
    asyncProcessing: false,
    teamCollaboration: false,
    advancedTables: false,
    captionEditor: false, 
    googleDocs: false,
    msWord: false,
    plagiarismCheck: false,
    templateCategories: ['Student', 'Content Creator', 'Others'],
    googleDriveExport: false,
    customTemplates: true
  },
  premium: {
    maxDocuments: 100,
    maxTemplates: 50,
    llmEnabled: true,
    multiLanguage: true,
    ocrLanguages: ['eng', 'fra', 'deu', 'spa', 'rus', 'ara', 'chi_sim', 'jpn', 'kor', 'hin'],
    styleGuideCompliance: true,
    asyncProcessing: true,
    teamCollaboration: true,
    advancedTables: true,
    captionEditor: true,
    googleDocs: true,
    msWord: true,
    plagiarismCheck: true,
    templateCategories: ['Student', 'Content Creator', 'Researcher', 'Business Professional',
                         'Multilingual User', 'Author', 'Collaborator', 'Project Manager'],
    googleDriveExport: true,
    customTemplates: true
  }
};

// Feature definitions
const featureDefinitions = {
  maxDocuments: {
    title: 'Document Limit',
    description: 'Maximum number of documents you can process',
    icon: 'description',
    premium: true
  },
  maxTemplates: {
    title: 'Template Limit',
    description: 'Maximum number of templates you can create',
    icon: 'dashboard_customize',
    premium: true
  },
  llmEnabled: {
    title: 'AI Enhancements',
    description: 'Advanced AI-powered text improvements',
    icon: 'auto_awesome',
    premium: true
  },
  multiLanguage: {
    title: 'Multi-language Support',
    description: 'Process documents in multiple languages',
    icon: 'translate',
    premium: true
  },
  styleGuideCompliance: {
    title: 'Style Guide Compliance',
    description: 'Ensure documents follow style guidelines',
    icon: 'style',
    premium: true
  },
  teamCollaboration: {
    title: 'Team Collaboration',
    description: 'Work together on documents',
    icon: 'group',
    premium: true
  },
  advancedTables: {
    title: 'Advanced Tables',
    description: 'Enhanced table formatting and editing',
    icon: 'table_chart',
    premium: true
  },
  plagiarismCheck: {
    title: 'Plagiarism Check',
    description: 'Verify document originality',
    icon: 'fact_check',
    premium: true
  }
};

export const SubscriptionProvider = ({ children }) => {
  const [userTier, setUserTier] = useState('free');
  const [isDevMode, setIsDevMode] = useState(false);
  
  useEffect(() => {
    getSubscriptionDetails();
  }, []);
  
  const getSubscriptionDetails = async () => {
    try {
      const userData = await getCurrentUser();
      if (userData) {
        setUserTier(userData.subscription || 'free');
        setIsDevMode(userData.isDevMode || false);
      }
    } catch (error) {
      console.error('Error fetching user subscription:', error);
    }
  };
  
  const canUseFeature = (featureKey) => {
    return tierFeatures[userTier][featureKey] || false;
  };
  
  const getFeatureLimit = (featureKey) => {
    return tierFeatures[userTier][featureKey] || 0;
  };
  
  const getFeatureDetails = (featureKey) => {
    return {
      ...featureDefinitions[featureKey],
      value: tierFeatures[userTier][featureKey],
      available: canUseFeature(featureKey)
    };
  };
  
  const canUseTemplate = (templateCategory) => {
    if (!templateCategory) return true;
    const allowedCategories = tierFeatures[userTier].templateCategories || [];
    return allowedCategories.includes(templateCategory);
  };
  
  // New function to switch between plans in dev mode
  const switchPlan = (planType) => {
    if (isDevMode && (planType === 'free' || planType === 'premium')) {
      setUserTier(planType);
      console.log(`Switched to ${planType} plan in dev mode`);
      return true;
    }
    return false;
  };
  
  const value = {
    userTier,
    canUseFeature,
    getFeatureLimit,
    getFeatureDetails,
    canUseTemplate,
    switchPlan,
    isDevMode,
    // For debugging
    features: tierFeatures[userTier],
    allFeatures: tierFeatures
  };
  
  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  );
};

export const useSubscription = () => useContext(SubscriptionContext);